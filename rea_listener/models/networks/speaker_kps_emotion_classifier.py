import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from einops import rearrange, reduce, repeat
import torch.nn.init as init
import sys
from .transformer.decoder import TransformerDecoder
from .transformer.encoder import TransformerEncoder, ConformerEncoder
from .att import AttentionBlock
from .utils.mask import make_non_pad_mask, subsequent_mask, make_pad_mask
import math
import matplotlib.pyplot as plt

class speaker_kps_based_emotion_classify_model(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        self.encoder_kps = TransformerEncoder(input_size=208, output_size=128, linear_units=512, input_layer='linear', num_blocks=2, dropout_rate=0.1, attention_heads=4)
        # self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.projector = nn.Linear(128, 128)
        
        self.classifier = nn.Linear(128, 7)

        self.loss_func = nn.CrossEntropyLoss()

    def get_loss(
        self,
        logits,
        emotion
    ):
        bs = logits.size(0)
        
        loss = self.loss_func(logits, emotion.squeeze(-1))

        with torch.no_grad():
            predicted = torch.argmax(logits, dim = -1)  # shape: (B,)
            step_acc = torch.sum((predicted == emotion.squeeze(-1)).float()).item() / bs
            # print(predicted[:10], emotion[:10])
            loss_dict = {
                'ce_loss': {
                    'val': loss.item(),
                    'n': bs,
                },
                'acc': {
                    'val': step_acc,
                    'n': bs,
                },
            }
            for key in loss_dict:
                if 'emotion' in key:
                    continue
                loss_dict[key]['weight'] = 1

        return loss, loss_dict


    def forward(
        self,
        audio,
        driven,
        init,
        lengths,
        epoch,
        change_epoch,
        target=None,
        emotion = None,
        speaker_kps = None,
        train = True
    ):
        if train :
            ########################## train
            logits = self.network_forward(speaker_kps, lengths)
            #print(logits[0])
            loss, loss_dict = self.get_loss(logits, emotion)

            return loss, loss_dict, logits
        

        else:
            start_pos, emotion_seq = self.pred_emotion_seq(speaker_kps)
            #print(start_pos, emotion_seq)
            return start_pos, emotion_seq
    
    def network_forward(self, speaker_kps, lengths):
        hidden_states, mask, pos = self.encoder_kps(speaker_kps, lengths)   
        #print(encoder_mask.shape, encoder_mask)    #(256,1,90) 有效为True，填充为False
        hidden_states = self.projector(hidden_states)

        mask_float = mask.float()  # shape: (b, 1, t)
        mask_expanded = mask_float.transpose(1, 2)  # shape: (b, t, 1)
        masked_tensor = hidden_states * mask_expanded  # shape: (b, t, d)
        sum_valid = masked_tensor.sum(dim=1)  # shape: (b, d)
        valid_counts = mask_float.sum(dim=-1)  # shape: (b, 1)

        epsilon = 1e-8
        mean_valid = sum_valid / (valid_counts + epsilon)  # shape: (b, d)
        logits = self.classifier(mean_valid)

        return logits

    
    def pred_emotion_seq(self, kps):
        b,t,d = kps.shape

        start_pos = list(range(0, t-90, 80))
        if t>90:
            start_pos.append(t-90)

        if len(start_pos) <= 0:
            start_pos.append(0)

        emotion_list = []
        with torch.no_grad():

            for start_frame in start_pos:
                input_kps = kps[:,start_frame: min(t, start_frame+90),:]
                lengths = lengths = torch.tensor([input_kps.shape[1]]).cuda()
                logits_step = self.network_forward(input_kps, lengths)
                #print(logits_step)
                pred = torch.argmax(logits_step, dim = -1).squeeze(0)
                
                emotion_list.append(pred)
        
        return start_pos, emotion_list





        
