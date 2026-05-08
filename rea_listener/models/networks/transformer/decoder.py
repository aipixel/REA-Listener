from typing import Tuple, List, Optional

import torch
import math

from .attention import MultiHeadedAttention
from .decoder_layer import DecoderLayer, DecoderLayerDoubleCrossAttention
from .embedding import PositionalEncoding
from .embedding import PeriodicPositionalEncoding
from .embedding import NoPositionalEncoding
from .embedding import SamePositionalEncoding
from .positionwise_feed_forward import PositionwiseFeedForward
from ..utils.mask import (subsequent_mask, make_pad_mask)

def init_biased_mask(n_head, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(n)) 
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask

class TransformerDecoder(torch.nn.Module):
    """Base class of Transfomer decoder module.
    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the hidden units number of position-wise feedforward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before:
            True: use layer_norm before each sub-block of a layer.
            False: use layer_norm after each sub-block of a layer.
        src_attention: if false, encoder-decoder cross attention is not
                       applied, such as CIF model
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        normalize_before: bool = True,
        src_attention: bool = True,
        use_style_layer = False
    ):
        super().__init__()
        attention_dim = encoder_output_size

        if input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(vocab_size, attention_dim),
                PositionalEncoding(attention_dim, positional_dropout_rate),
            )
        elif input_layer == 'ped_pos':
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(vocab_size, attention_dim),
                PeriodicPositionalEncoding(attention_dim, positional_dropout_rate),
            )
        elif input_layer == 'none':
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(vocab_size, attention_dim),
                NoPositionalEncoding(attention_dim, positional_dropout_rate),
            )
        elif input_layer == 'same':
            self.embed1 = torch.nn.Linear(vocab_size, attention_dim)
            self.embed2 = SamePositionalEncoding(attention_dim, positional_dropout_rate)
            #self.embed = NoPositionalEncoding(attention_dim,
#                                              positional_dropout_rate)
        else:
            raise ValueError(f"only 'embed' is supported: {input_layer}")

        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(attention_dim, eps=1e-5)
        self.use_output_layer = use_output_layer
        self.output_layer = torch.nn.Linear(attention_dim, vocab_size)
        self.num_blocks = num_blocks
        self.decoders = torch.nn.ModuleList([
            DecoderLayer(
                attention_dim,
                MultiHeadedAttention(attention_heads, attention_dim,
                                     self_attention_dropout_rate),
                MultiHeadedAttention(attention_heads, attention_dim,
                                     src_attention_dropout_rate)
                if src_attention else None,
                PositionwiseFeedForward(attention_dim, linear_units,
                                        dropout_rate),
                dropout_rate,
                normalize_before,
            ) for _ in range(self.num_blocks)
        ])

        self.use_style_layer = use_style_layer
        if self.use_style_layer:
            self.style_layers  = torch.nn.ModuleList([
                StylizationBlock(
                    latent_dim = attention_dim,
                    time_embed_dim = attention_dim,
                    dropout=dropout_rate
                ) for _ in range(self.num_blocks)
            ])

    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        epoch: int = 101,
        change_epoch: int = 100,
        r_ys_in_pad: torch.Tensor = torch.empty(0),
        reverse_weight: float = 0.0,
        emotion_embed = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward decoder.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoder memory mask, (batch, 1, maxlen_in)
            ys_in_pad: padded input token ids, int64 (batch, maxlen_out)
            ys_in_lens: input lengths of this batch (batch)
            r_ys_in_pad: not used in transformer decoder, in order to unify api
                with bidirectional decoder
            reverse_weight: not used in transformer decoder, in order to unify
                api with bidirectional decode
        Returns:
            (tuple): tuple containing:
                x: decoded token score before softmax (batch, maxlen_out,
                    vocab_size) if use_output_layer is True,
                torch.tensor(0.0), in order to unify api with bidirectional decoder
                olens: (batch, )
        """
        tgt = ys_in_pad
        maxlen = tgt.size(1)
        # tgt_mask: (B, 1, L)
        tgt_mask = ~make_pad_mask(ys_in_lens, maxlen).unsqueeze(1)
        tgt_mask = tgt_mask.to(tgt.device)
        # m: (1, L, L)
        if epoch <= change_epoch:
            m = subsequent_mask(tgt_mask.size(-1),
                                device=tgt_mask.device).unsqueeze(0)
        else:
            m = torch.ones(tgt_mask.size(-1), tgt_mask.size(-1)).unsqueeze(0).int().to(tgt.device)
        #m = init_biased_mask(4, 1000, 1).repeat(tgt.size(0), 1, 1)
        #m = init_biased_mask(4, 5000, 10)
        #m = m[:, :tgt.shape[1], :tgt.shape[1]].clone().detach().to(tgt.device)
        #print(m.shape)
        # tgt_mask: (B, L, L)
        tgt_mask = tgt_mask & m
        #tgt_mask = (tgt_mask, m)
        #x = self.embed1(tgt)
        #x, _ = self.embed2(x, pos)
        x, _ = self.embed(tgt)
        for idx, layer in enumerate(self.decoders):
            # print("tgt_mask:",tgt_mask.shape)
            # print(tgt_mask)
            # print("memory_mask:",memory_mask.shape)
            # print(memory_mask)
            # exit()

            x, tgt_mask, memory, memory_mask = layer(x, tgt_mask, memory,
                                                     memory_mask)
            if self.use_style_layer:
                x = self.style_layers[idx](x, emotion_embed)

        if self.normalize_before:
            x = self.after_norm(x)
        if self.use_output_layer:
            x = self.output_layer(x)
        #olens = tgt_mask.sum(1)
        olens = 1
        return x, torch.tensor(0.0), olens

    def forward_one_step(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        cache: Optional[List[torch.Tensor]] = None,
        emotion_embed = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.
            This is only used for decoding.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoded memory mask, (batch, 1, maxlen_in)
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            cache: cached output list of (batch, max_time_out-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        assert False
        x, _ = self.embed(tgt)
        new_cache = []
        for i, decoder in enumerate(self.decoders):
            if cache is None:
                c = None
            else:
                c = cache[i]
            x, tgt_mask, memory, memory_mask = decoder(x,
                                                       tgt_mask,
                                                       memory,
                                                       memory_mask,
                                                       cache=c)
            new_cache.append(x)
        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        if self.use_output_layer:
            y = torch.log_softmax(self.output_layer(y), dim=-1)
        return y, new_cache

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class StylizationBlock(torch.nn.Module):
    def __init__(self, latent_dim, time_embed_dim, dropout):
        super().__init__()
        self.emb_layers = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(time_embed_dim, 2 * latent_dim),
        )
        self.norm = torch.nn.LayerNorm(latent_dim)
        self.out_layers = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Dropout(p=dropout),
            zero_module(torch.nn.Linear(latent_dim, latent_dim)),
        )

    def forward(self, h, emb):
        """
        h: B, T, D
        emb: B, D
        """
        # B, 1, 2D
        emb_out = self.emb_layers(emb).unsqueeze(1)
        # scale: B, 1, D / shift: B, 1, D
        scale, shift = torch.chunk(emb_out, 2, dim=2)
        h = self.norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return h


class TransformerDecoderDoubleCrossAttention(torch.nn.Module):
    """Base class of Transfomer decoder module.
    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the hidden units number of position-wise feedforward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before:
            True: use layer_norm before each sub-block of a layer.
            False: use layer_norm after each sub-block of a layer.
        src_attention: if false, encoder-decoder cross attention is not
                       applied, such as CIF model
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        normalize_before: bool = True,
        src_attention: bool = True,
    ):
        super().__init__()
        attention_dim = encoder_output_size

        if input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(vocab_size, attention_dim),
                PositionalEncoding(attention_dim, positional_dropout_rate),
            )
        elif input_layer == 'ped_pos':
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(vocab_size, attention_dim),
                PeriodicPositionalEncoding(attention_dim, positional_dropout_rate),
            )
        elif input_layer == 'none':
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(vocab_size, attention_dim),
                NoPositionalEncoding(attention_dim, positional_dropout_rate),
            )
        elif input_layer == 'same':
            self.embed1 = torch.nn.Linear(vocab_size, attention_dim)
            self.embed2 = SamePositionalEncoding(attention_dim, positional_dropout_rate)
            #self.embed = NoPositionalEncoding(attention_dim,
#                                              positional_dropout_rate)
        else:
            raise ValueError(f"only 'embed' is supported: {input_layer}")

        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(attention_dim, eps=1e-5)
        self.use_output_layer = use_output_layer
        self.output_layer = torch.nn.Linear(attention_dim, vocab_size)
        self.num_blocks = num_blocks
        self.decoders = torch.nn.ModuleList([
            DecoderLayerDoubleCrossAttention(
                attention_dim,
                MultiHeadedAttention(attention_heads, attention_dim,
                                     self_attention_dropout_rate),
                MultiHeadedAttention(attention_heads, attention_dim,
                                     src_attention_dropout_rate)
                if src_attention else None,
                MultiHeadedAttention(attention_heads, attention_dim,
                                     src_attention_dropout_rate)
                if src_attention else None,
                PositionwiseFeedForward(attention_dim, linear_units,
                                        dropout_rate),
                dropout_rate,
                normalize_before,
            ) for _ in range(self.num_blocks)
        ])

    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        epoch: int = 101,
        change_epoch: int = 100,
        r_ys_in_pad: torch.Tensor = torch.empty(0),
        reverse_weight: float = 0.0,

        memory2: torch.Tensor = None,
        memory_mask2: torch.Tensor = None,


    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward decoder.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoder memory mask, (batch, 1, maxlen_in)
            ys_in_pad: padded input token ids, int64 (batch, maxlen_out)
            ys_in_lens: input lengths of this batch (batch)
            r_ys_in_pad: not used in transformer decoder, in order to unify api
                with bidirectional decoder
            reverse_weight: not used in transformer decoder, in order to unify
                api with bidirectional decode
        Returns:
            (tuple): tuple containing:
                x: decoded token score before softmax (batch, maxlen_out,
                    vocab_size) if use_output_layer is True,
                torch.tensor(0.0), in order to unify api with bidirectional decoder
                olens: (batch, )
        """
        tgt = ys_in_pad
        maxlen = tgt.size(1)
        # tgt_mask: (B, 1, L)
        tgt_mask = ~make_pad_mask(ys_in_lens, maxlen).unsqueeze(1)
        tgt_mask = tgt_mask.to(tgt.device)
        # m: (1, L, L)
        if epoch <= change_epoch:
            m = subsequent_mask(tgt_mask.size(-1),
                                device=tgt_mask.device).unsqueeze(0)
        else:
            m = torch.ones(tgt_mask.size(-1), tgt_mask.size(-1)).unsqueeze(0).int().to(tgt.device)
        #m = init_biased_mask(4, 1000, 1).repeat(tgt.size(0), 1, 1)
        #m = init_biased_mask(4, 5000, 10)
        #m = m[:, :tgt.shape[1], :tgt.shape[1]].clone().detach().to(tgt.device)
        #print(m.shape)
        # tgt_mask: (B, L, L)
        tgt_mask = tgt_mask & m
        #tgt_mask = (tgt_mask, m)
        #x = self.embed1(tgt)
        #x, _ = self.embed2(x, pos)
        x, _ = self.embed(tgt)
        for layer in self.decoders:
            # print("tgt_mask:",tgt_mask.shape)
            # print(tgt_mask)
            # print("memory_mask:",memory_mask.shape)
            # print(memory_mask)
            # exit()

            x, tgt_mask, memory, memory_mask, memory2, memory_mask2 = layer(x, tgt_mask, memory,
                                                     memory_mask,
                                                     memory2, 
                                                     memory_mask2)
        if self.normalize_before:
            x = self.after_norm(x)
        if self.use_output_layer:
            x = self.output_layer(x)
        #olens = tgt_mask.sum(1)
        olens = 1
        return x, torch.tensor(0.0), olens

    def forward_one_step(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        cache: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.
            This is only used for decoding.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoded memory mask, (batch, 1, maxlen_in)
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            cache: cached output list of (batch, max_time_out-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        x, _ = self.embed(tgt)
        new_cache = []
        for i, decoder in enumerate(self.decoders):
            if cache is None:
                c = None
            else:
                c = cache[i]
            x, tgt_mask, memory, memory_mask = decoder(x,
                                                       tgt_mask,
                                                       memory,
                                                       memory_mask,
                                                       cache=c)
            new_cache.append(x)
        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        if self.use_output_layer:
            y = torch.log_softmax(self.output_layer(y), dim=-1)
        return y, new_cache



class BiTransformerDecoder(torch.nn.Module):
    """Base class of Transfomer decoder module.
    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the hidden units number of position-wise feedforward
        num_blocks: the number of decoder blocks
        r_num_blocks: the number of right to left decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before:
            True: use layer_norm before each sub-block of a layer.
            False: use layer_norm after each sub-block of a layer.
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        r_num_blocks: int = 0,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        normalize_before: bool = True,
    ):

        super().__init__()
        self.left_decoder = TransformerDecoder(
            vocab_size, encoder_output_size, attention_heads, linear_units,
            num_blocks, dropout_rate, positional_dropout_rate,
            self_attention_dropout_rate, src_attention_dropout_rate,
            input_layer, use_output_layer, normalize_before)

        self.right_decoder = TransformerDecoder(
            vocab_size, encoder_output_size, attention_heads, linear_units,
            r_num_blocks, dropout_rate, positional_dropout_rate,
            self_attention_dropout_rate, src_attention_dropout_rate,
            input_layer, use_output_layer, normalize_before)

    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        r_ys_in_pad: torch.Tensor,
        reverse_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward decoder.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoder memory mask, (batch, 1, maxlen_in)
            ys_in_pad: padded input token ids, int64 (batch, maxlen_out)
            ys_in_lens: input lengths of this batch (batch)
            r_ys_in_pad: padded input token ids, int64 (batch, maxlen_out),
                used for right to left decoder
            reverse_weight: used for right to left decoder
        Returns:
            (tuple): tuple containing:
                x: decoded token score before softmax (batch, maxlen_out,
                    vocab_size) if use_output_layer is True,
                r_x: x: decoded token score (right to left decoder)
                    before softmax (batch, maxlen_out, vocab_size)
                    if use_output_layer is True,
                olens: (batch, )
        """
        l_x, _, olens = self.left_decoder(memory, memory_mask, ys_in_pad,
                                          ys_in_lens)
        r_x = torch.tensor(0.0)
        if reverse_weight > 0.0:
            r_x, _, olens = self.right_decoder(memory, memory_mask, r_ys_in_pad,
                                               ys_in_lens)
        return l_x, r_x, olens

    def forward_one_step(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        cache: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.
            This is only used for decoding.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoded memory mask, (batch, 1, maxlen_in)
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            cache: cached output list of (batch, max_time_out-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        """
        return self.left_decoder.forward_one_step(memory, memory_mask, tgt,
                                                  tgt_mask, cache)
