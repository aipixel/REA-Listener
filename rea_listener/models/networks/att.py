from collections import defaultdict
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from PIL import Image

from einops import rearrange, repeat

try:
    from panopticapi.utils import id2rgb, rgb2id
except ImportError:
    pass


class AttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, sr_ratio=1,
                 mask_all_motion = False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        # for downsample
        self.norm = nn.LayerNorm(d_model)

        self.mask_all_motion = mask_all_motion

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):

        if memory is not None:
            memory = self.norm(memory)

        b = tgt.size(1)
        # self attn
        q = k = self.with_pos_embed(tgt, pos)
        v = tgt
        #print("tgt_key_padding_mask:", tgt_key_padding_mask.shape)
        #print(tgt_key_padding_mask)
        tgt2 = self.self_attn(q, k, value=v, attn_mask=None,
                              key_padding_mask=tgt_key_padding_mask)[0] # [H*W, B*T, C]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attn
        if self.mask_all_motion:
            # actually self-attention
            q = k = self.with_pos_embed(tgt, query_pos)
            v = tgt
            tgt2 = self.self_attn(q, k, value=v, attn_mask=None,
                              key_padding_mask=tgt_key_padding_mask)[0] # [H*W, B*T, C]

        else:
            #print("memory:", memory.shape)
            tgt2 = self.multihead_attn(query=self.with_pos_embed(memory, query_pos),
                                    key=self.with_pos_embed(tgt, pos),
                                    value=tgt, attn_mask=None,
                                    key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


    def forward(self, tgt, memory,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        return self.forward_post(tgt, memory,
                                 tgt_key_padding_mask, memory_key_padding_mask, 
                                 pos, query_pos)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


