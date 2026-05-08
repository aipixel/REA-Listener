from numpy import percentile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from einops import rearrange, reduce, repeat
import torch.nn.init as init
import sys
from .transformer.decoder import TransformerDecoder
from .transformer.encoder import TransformerEncoder, ConformerEncoder, BaseEncoder
from .transformer.attention import MultiHeadedAttention

from .att import AttentionBlock
from .utils.mask import make_non_pad_mask, subsequent_mask, make_pad_mask
import math
import matplotlib.pyplot as plt
import random
from typing import Tuple, Optional, Literal
from dataclasses import dataclass
import torch.distributed as dist
import numpy as np
import time


class AudioEncoder(nn.Module):
    def __init__(
        self,
        moe_args
    ):
        super().__init__()
        
        self.block1 = TransformerEncoder(input_size=41, output_size=256, linear_units=1024, input_layer='linear', num_blocks=1, dropout_rate=0.1, attention_heads=4)
        self.block2 = MOETransformerEncoder(input_size=256, output_size=256, linear_units=1024, input_layer='linear', num_blocks=2, dropout_rate=0.1, attention_heads=4,
                                            moe_args = moe_args)
    
    def forward(self, x, x_lengths, moe_condition):
        y, y_mask, pos = self.block1(x, x_lengths)
        y, y_mask, pos = self.block2(y, x_lengths, moe_condition=moe_condition)
        return y, y_mask, pos

class KpsEncoder(nn.Module):
    def __init__(
        self,
        moe_args
    ):
        super().__init__()
        
        self.block1 = TransformerEncoder(input_size=208, output_size=256, linear_units=1024, input_layer='linear', num_blocks=1, dropout_rate=0.1, attention_heads=4)
        self.block2 = MOETransformerEncoder(input_size=256, output_size=256, linear_units=1024, input_layer='linear', num_blocks=2, dropout_rate=0.1, attention_heads=4,
                                            moe_args = moe_args)
    
    def forward(self, x, x_lengths, moe_condition):
        y, y_mask, pos = self.block1(x, x_lengths)
        y, y_mask, pos = self.block2(y, x_lengths, moe_condition=moe_condition)
        return y, y_mask, pos

class REAListener(nn.Module):
    def __init__(
        self,
        param,    
        emotion_classify_model = None,
        balance_moe = False,
        train = False,
        decoder_num_blocks = 3,
    ):
        super().__init__()
        generator_cfg = param.model.generator
        self.generator_cfg = generator_cfg
        self.loss_weights = param.loss_weights
        self.loss_weights['TOTAL_LOSS'] = 1
        self.loss_names = ['TOTAL_LOSS'] + sorted(list(self.loss_weights.keys()))
        #self.dynam_3dmm_split = [3, 64, 3, 3]
        self.dynam_3dmm_split = [100, 9, 6]

        moe_args = ModelArgs()

        if balance_moe:
            print("balance moe")
            moe_args.score_func = "sigmoid"
            moe_args.use_bias = True

            if not train:
                # 测试时不要更新bias
                moe_args.bias_update_rate = -1.0



        self.audio_encoder = AudioEncoder(moe_args)
        self.kps_encoder = KpsEncoder(moe_args)

        self.decoder = TransformerDecoder(vocab_size=115, encoder_output_size=256+64, num_blocks=decoder_num_blocks, linear_units=1024, input_layer='embed', use_output_layer=True, dropout_rate=0.1, attention_heads=4)
        self.emotion_embed = nn.Embedding(7, 64)
        self.modal_type_embed_layer = nn.Embedding(3, 64)


        self.emotion_loss_weight = 0.0
        self.emotion_classify_model = None
        if param.emotion_loss_weight > 0:
            if emotion_classify_model is None:
                raise ValueError(
                    "emotion_loss_weight > 0 requires an explicit emotion_classify_model for "
                    "REAListener."
                )
            self.emotion_loss_weight = param.emotion_loss_weight
            self.emotion_classify_model = emotion_classify_model
            for p in self.emotion_classify_model.parameters():
                p.requires_grad = False

        
    def predict(
        self,
        audio,
        driven,
        init,
        lengths,
        epoch,
        change_epoch,
        target,
        emotion,
        speaker_kps,
        modal_type,
        lengths_kps,
    ):
        if modal_type is None:
            raise ValueError("modal_type must be provided explicitly for REAListener.predict(). Use 0=audio, 1=visual, 2=audio-visual.")
        if lengths_kps is None:
            raise ValueError("lengths_kps must be provided explicitly for REAListener.predict().")
        if len(modal_type.shape) > 1:
            modal_type = modal_type.squeeze(1)

        encoder_out, encoder_mask = self.fusion_encode(audio, lengths, speaker_kps, lengths_kps, modal_type)

        b,t,d = encoder_out.shape
        emotion_feature = self.emotion_embed(emotion) #(b,1,d)
        encoder_out = torch.cat([encoder_out, emotion_feature.repeat(1,t,1)], dim=-1)
        init = init.unsqueeze(1)

        ## teacher_forcing

        if epoch <= change_epoch:
            target = torch.cat((init, target), 1)
        else:
            target = init.repeat(1, target.shape[1]+1,  1)
        decoder_out, _, _ = self.decoder(encoder_out, encoder_mask, target, lengths, epoch, change_epoch)
        return decoder_out

    def decode_period_sl_initlast_cat(
        self,
        audio,
        driven,
        init,
        lengths,
        change_epoch = 100,
        emotion_list = None,
        speaker_kps = None,
        encode_type = 'kps',
    ):
        assert encode_type in ['kps', 'audio', 'together']

        epoch = 500
        period = 90
        slide = 80 
        frame_nums = audio.shape[1]
        #rounds = (frame_nums - 1) // slide + 1
        init = init.unsqueeze(1)
        current = 0
        segment_count = 0
        while (current + period) < frame_nums :
            audio_cat = audio[:, int(current):int(current + period), :]
            kps_cat = speaker_kps[:, int(current):int(current + period), :]

            if encode_type == 'kps':
                lengths = torch.tensor([0]).cuda().long()
                lengths_kps = torch.tensor([period]).cuda().long()
                modal_type = torch.tensor([0]).cuda().long()

            elif encode_type == 'audio':
                lengths = torch.tensor([period]).cuda().long()
                lengths_kps = torch.tensor([0]).cuda().long()
                modal_type = torch.tensor([1]).cuda().long()

            elif encode_type == 'together':
                lengths = torch.tensor([period]).cuda().long()
                lengths_kps = torch.tensor([period]).cuda().long()
                modal_type = torch.tensor([2]).cuda().long()
            encoder_out, encoder_mask = self.fusion_encode(audio_cat, lengths, kps_cat, lengths_kps, modal_type)
            b,t,d = encoder_out.shape
            emotion_feature = self.emotion_embed(emotion_list[segment_count]) #(b,1,d)
            segment_count += 1
            encoder_out = torch.cat([encoder_out, emotion_feature.repeat(1,t,1)], dim=-1)
            


            frame_num = encoder_out.shape[1]
            if current == 0:
                decoder_in = init.repeat(1, frame_num,  1)
            else:
                decoder_in = decoder_out[:, -(period - slide), :].unsqueeze(1).repeat(1, frame_num,  1)

            d_length = frame_num
            d_length = torch.tensor([d_length])
            decoder_in, _, _ = self.decoder(encoder_out, encoder_mask, decoder_in, d_length, epoch, change_epoch)
            if current == 0: 
                decoder_out = decoder_in
            else: 
                decoder_out = decoder_out[:, :-(period - slide),:]
                decoder_out = torch.cat([decoder_out, decoder_in], 1)
            current = current + slide


        if current != 0:
            # last segment
            audio_cat = audio[:, -period:, :]
            kps_cat = speaker_kps[:, -period:, :]

            if encode_type == 'kps':
                lengths = torch.tensor([0]).cuda().long()
                lengths_kps = torch.tensor([period]).cuda().long()
                modal_type = torch.tensor([0]).cuda().long()

            elif encode_type == 'audio':
                lengths = torch.tensor([period]).cuda().long()
                lengths_kps = torch.tensor([0]).cuda().long()
                modal_type = torch.tensor([1]).cuda().long()

            elif encode_type == 'together':
                lengths = torch.tensor([period]).cuda().long()
                lengths_kps = torch.tensor([period]).cuda().long()
                modal_type = torch.tensor([2]).cuda().long()
            encoder_out, encoder_mask = self.fusion_encode(audio_cat, lengths, kps_cat, lengths_kps, modal_type)
            driven_frame = audio.shape[1] - period - 1

            b,t,d = encoder_out.shape
            emotion_feature = self.emotion_embed(emotion_list[segment_count]) #(b,1,d)
            segment_count += 1
            encoder_out = torch.cat([encoder_out, emotion_feature.repeat(1,t,1)], dim=-1)




            frame_num = encoder_out.shape[1]
            
            decoder_in = decoder_out[:, driven_frame, :].unsqueeze(1).repeat(1, frame_num,  1)
            #decoder_in = init_all
            #print(frame_num)
            d_length = frame_num
            #print(d_length)
            d_length = torch.tensor([d_length])
            decoder_in, _, _ = self.decoder(encoder_out, encoder_mask, decoder_in, d_length, epoch, 100)    #change_epoch)
            #print(frame_num)

            ####
            decoder_out = decoder_out[:, :driven_frame+1,:]
            decoder_out = torch.cat([decoder_out, decoder_in], 1)
            #decoder_in_right = decoder_in[:, int(period +decoder_out.shape[1] - frame_nums):, :]
            #decoder_out = torch.cat((decoder_out, decoder_in_right), 1)

        else:
            # short video small than period          
            audio_cat = audio
            kps_cat = speaker_kps

            if encode_type == 'kps':
                lengths = torch.tensor([0]).cuda().long()
                lengths_kps = torch.tensor([audio.shape[1]]).cuda().long()
                modal_type = torch.tensor([0]).cuda().long()

            elif encode_type == 'audio':
                lengths = torch.tensor([audio.shape[1]]).cuda().long()
                lengths_kps = torch.tensor([0]).cuda().long()
                modal_type = torch.tensor([1]).cuda().long()

            elif encode_type == 'together':
                lengths = torch.tensor([audio.shape[1]]).cuda().long()
                lengths_kps = torch.tensor([audio.shape[1]]).cuda().long()
                modal_type = torch.tensor([2]).cuda().long()
            encoder_out, encoder_mask = self.fusion_encode(audio_cat, lengths, kps_cat, lengths_kps, modal_type)
            b,t,d = encoder_out.shape
            emotion_feature = self.emotion_embed(emotion_list[segment_count]) #(b,1,d)
            segment_count += 1
            encoder_out = torch.cat([encoder_out, emotion_feature.repeat(1,t,1)], dim=-1)

            frame_num = encoder_out.shape[1]
            decoder_in = init.repeat(1, frame_num,  1)

            d_length = frame_num

            d_length = torch.tensor([d_length])
            decoder_in, _, _ = self.decoder(encoder_out, encoder_mask, decoder_in, d_length, epoch, 100)    #change_epoch)

            decoder_out = decoder_in

        assert decoder_out.shape[1] == audio.shape[1] 
        return decoder_out

    
    def decode_period_sl_initlast_cat_for_speed(
        self,
        audio,
        driven,
        init,
        lengths,
        change_epoch = 100,
        emotion_list = None,
        speaker_kps = None,
        encode_type = 'kps',
    ):
        total_enc_time = 0
        total_dec_time = 0
        assert encode_type in ['kps', 'audio', 'together']

        epoch = 500
        period = 90
        slide = 80 
        frame_nums = audio.shape[1]
        #rounds = (frame_nums - 1) // slide + 1
        init = init.unsqueeze(1)
        current = 0
        segment_count = 0
        while (current + period) < frame_nums :
            audio_cat = audio[:, int(current):int(current + period), :]
            kps_cat = speaker_kps[:, int(current):int(current + period), :]

            if encode_type == 'kps':
                lengths = torch.tensor([0]).cuda().long()
                lengths_kps = torch.tensor([period]).cuda().long()
                modal_type = torch.tensor([0]).cuda().long()

            elif encode_type == 'audio':
                lengths = torch.tensor([period]).cuda().long()
                lengths_kps = torch.tensor([0]).cuda().long()
                modal_type = torch.tensor([1]).cuda().long()

            elif encode_type == 'together':
                lengths = torch.tensor([period]).cuda().long()
                lengths_kps = torch.tensor([period]).cuda().long()
                modal_type = torch.tensor([2]).cuda().long()
            encoder_out, encoder_mask, enc_time = self.fusion_encode_for_speed(audio_cat, lengths, kps_cat, lengths_kps, modal_type)
            total_enc_time += enc_time
            b,t,d = encoder_out.shape

            time_before_decode = time.time()
            emotion_feature = self.emotion_embed(emotion_list[segment_count]) #(b,1,d)
            segment_count += 1
            encoder_out = torch.cat([encoder_out, emotion_feature.repeat(1,t,1)], dim=-1)
            


            frame_num = encoder_out.shape[1]
            if current == 0:
                decoder_in = init.repeat(1, frame_num,  1)
            else:
                decoder_in = decoder_out[:, -(period - slide), :].unsqueeze(1).repeat(1, frame_num,  1)

            d_length = frame_num
            d_length = torch.tensor([d_length])
            decoder_in, _, _ = self.decoder(encoder_out, encoder_mask, decoder_in, d_length, epoch, change_epoch)
            if current == 0: 
                decoder_out = decoder_in
            else: 
                decoder_out = decoder_out[:, :-(period - slide),:]
                decoder_out = torch.cat([decoder_out, decoder_in], 1)
            current = current + slide
            time_after_decode = time.time()
            total_dec_time += time_after_decode-time_before_decode

        if current != 0:
            # last segment
            audio_cat = audio[:, -period:, :]
            kps_cat = speaker_kps[:, -period:, :]

            if encode_type == 'kps':
                lengths = torch.tensor([0]).cuda().long()
                lengths_kps = torch.tensor([period]).cuda().long()
                modal_type = torch.tensor([0]).cuda().long()

            elif encode_type == 'audio':
                lengths = torch.tensor([period]).cuda().long()
                lengths_kps = torch.tensor([0]).cuda().long()
                modal_type = torch.tensor([1]).cuda().long()

            elif encode_type == 'together':
                lengths = torch.tensor([period]).cuda().long()
                lengths_kps = torch.tensor([period]).cuda().long()
                modal_type = torch.tensor([2]).cuda().long()
            encoder_out, encoder_mask, enc_time = self.fusion_encode_for_speed(audio_cat, lengths, kps_cat, lengths_kps, modal_type)
            total_enc_time += enc_time
            time_before_decode = time.time()
            driven_frame = audio.shape[1] - period - 1

            b,t,d = encoder_out.shape
            emotion_feature = self.emotion_embed(emotion_list[segment_count]) #(b,1,d)
            segment_count += 1
            encoder_out = torch.cat([encoder_out, emotion_feature.repeat(1,t,1)], dim=-1)




            frame_num = encoder_out.shape[1]
            
            decoder_in = decoder_out[:, driven_frame, :].unsqueeze(1).repeat(1, frame_num,  1)
            #decoder_in = init_all
            #print(frame_num)
            d_length = frame_num
            #print(d_length)
            d_length = torch.tensor([d_length])
            decoder_in, _, _ = self.decoder(encoder_out, encoder_mask, decoder_in, d_length, epoch, 100)    #change_epoch)
            #print(frame_num)

            ####
            decoder_out = decoder_out[:, :driven_frame+1,:]
            decoder_out = torch.cat([decoder_out, decoder_in], 1)
            #decoder_in_right = decoder_in[:, int(period +decoder_out.shape[1] - frame_nums):, :]
            #decoder_out = torch.cat((decoder_out, decoder_in_right), 1)
            time_after_decode = time.time()
            total_dec_time += time_after_decode-time_before_decode

        else:
            # short video small than period          
            audio_cat = audio
            kps_cat = speaker_kps

            if encode_type == 'kps':
                lengths = torch.tensor([0]).cuda().long()
                lengths_kps = torch.tensor([audio.shape[1]]).cuda().long()
                modal_type = torch.tensor([0]).cuda().long()

            elif encode_type == 'audio':
                lengths = torch.tensor([audio.shape[1]]).cuda().long()
                lengths_kps = torch.tensor([0]).cuda().long()
                modal_type = torch.tensor([1]).cuda().long()

            elif encode_type == 'together':
                lengths = torch.tensor([audio.shape[1]]).cuda().long()
                lengths_kps = torch.tensor([audio.shape[1]]).cuda().long()
                modal_type = torch.tensor([2]).cuda().long()
            encoder_out, encoder_mask, enc_time = self.fusion_encode_for_speed(audio_cat, lengths, kps_cat, lengths_kps, modal_type)
            total_enc_time += enc_time
            time_before_decode = time.time()
            b,t,d = encoder_out.shape
            emotion_feature = self.emotion_embed(emotion_list[segment_count]) #(b,1,d)
            segment_count += 1
            encoder_out = torch.cat([encoder_out, emotion_feature.repeat(1,t,1)], dim=-1)

            frame_num = encoder_out.shape[1]
            decoder_in = init.repeat(1, frame_num,  1)

            d_length = frame_num

            d_length = torch.tensor([d_length])
            decoder_in, _, _ = self.decoder(encoder_out, encoder_mask, decoder_in, d_length, epoch, 100)    #change_epoch)

            decoder_out = decoder_in
            time_after_decode = time.time()
            total_dec_time += time_after_decode-time_before_decode

        assert decoder_out.shape[1] == audio.shape[1] 
        return decoder_out, total_enc_time, total_dec_time

    def get_3dmm_loss(self, pred, gt):
        b, t, c = pred.shape
        xpred = pred.view(b * t, c)
        xgt = gt.view(b * t, c)
        pairwise_distance = F.pairwise_distance(xpred, xgt)
        loss = torch.mean(pairwise_distance)
        spiky_loss = self.get_spiky_loss(pred, gt)
        return loss, spiky_loss
    
    def get_spiky_loss(self, pred, gt):
        b, t, c = pred.shape
        pred_spiky = pred[:, 1:, :] - pred[:, :-1, :]
        gt_spiky = gt[:, 1:, :] - gt[:, :-1, :]
        pred_spiky = pred_spiky.view(b * (t - 1), c)
        gt_spiky = gt_spiky.view(b * (t - 1), c)
        pairwise_distance = F.pairwise_distance(pred_spiky, gt_spiky)
        return torch.mean(pairwise_distance)

    def get_loss(
        self,
        pred_3dmm_dynam,
        oth_listener_3dmm_dynam,
        emotion = None
    ):
        bs = pred_3dmm_dynam.size(0)
        # angle / exp / trans loss
        #pd_angle, pd_exp, pd_trans, pd_crop = torch.split(pred_3dmm_dynam, self.dynam_3dmm_split, dim=-1)
        #gt_angle, gt_exp, gt_trans, gt_crop = torch.split(oth_listener_3dmm_dynam, self.dynam_3dmm_split, dim=-1)
        pd_exp, pd_pose_eye, pd_trans = torch.split(pred_3dmm_dynam, self.dynam_3dmm_split, dim=-1)
        gt_exp, gt_pose_eye, gt_trans = torch.split(oth_listener_3dmm_dynam, self.dynam_3dmm_split, dim=-1)

        angle_loss, angle_spiky_loss = self.get_3dmm_loss(pd_pose_eye, gt_pose_eye)
        exp_loss, exp_spiky_loss = self.get_3dmm_loss(pd_exp, gt_exp)
        trans_loss, trans_spiky_loss = self.get_3dmm_loss(pd_trans, gt_trans)

        # angle_loss, angle_spiky_loss = self.get_3dmm_loss(pd_angle, gt_angle)
        # exp_loss, exp_spiky_loss = self.get_3dmm_loss(pd_exp, gt_exp)
        # trans_loss, trans_spiky_loss = self.get_3dmm_loss(pd_trans, gt_trans)
        # crop_loss, crop_spiky_loss = self.get_3dmm_loss(pd_crop, gt_crop)

        loss = angle_loss       * self.loss_weights['loss_angle'] + \
               angle_spiky_loss * self.loss_weights['loss_angle_spiky'] + \
               exp_loss         * self.loss_weights['loss_exp'] + \
               exp_spiky_loss   * self.loss_weights['loss_exp_spiky'] + \
               trans_loss       * self.loss_weights['loss_trans'] + \
               trans_spiky_loss * self.loss_weights['loss_trans_spiky'] #+ \

        loss_emotion = pred_3dmm_dynam.new_tensor(0.0)
        loss_emotion_dict = {"Acc": {"val": 0.0}}
        if self.emotion_loss_weight > 0:
            if emotion is None:
                raise ValueError(
                    "emotion must be provided when emotion_loss_weight > 0 for "
                    "REAListener.get_loss()."
                )
            b, t, d = pred_3dmm_dynam.shape
            loss_emotion, loss_emotion_dict, _ = self.emotion_classify_model(
                pred_3dmm_dynam.view(b * t, d),
                emotion.repeat(1, t).view(-1),
            )
            loss = loss + loss_emotion * self.emotion_loss_weight

        with torch.no_grad():
            loss_dict = {
                'TOTAL_LOSS': {
                    'val': loss.item(),
                    'n': bs,
                },
                'loss_angle': {
                    'val': angle_loss.item(),
                    'n': bs,
                },
                'loss_angle_spiky': {
                    'val': angle_spiky_loss.item(),
                    'n': bs,
                },
                'loss_exp': {
                    'val': exp_loss.item(),
                    'n': bs,
                },
                'loss_exp_spiky': {
                    'val': exp_spiky_loss.item(),
                    'n': bs,
                },
                'loss_trans': {
                    'val': trans_loss.item(),
                    'n': bs,
                },
                'loss_trans_spiky': {
                    'val': trans_spiky_loss.item(),
                    'n': bs,
                },

                'loss_emotion': {
                    'val': loss_emotion.item(),
                    'weight': self.emotion_loss_weight,
                    'n': bs,
                },
                'acc_emotion': {
                    'val': loss_emotion_dict["Acc"]["val"],
                    'weight': 1,
                    'n': bs,
                },

            }
            for key in loss_dict:
                if 'emotion' in key:
                    continue
                loss_dict[key]['weight'] = self.loss_weights[key]
        return loss, loss_dict

    def fusion_encode(self, audio, audio_lengths, kps, kps_lengths, modal_type):
        b,t,_ = audio.shape

        assert len(modal_type.shape) == 1

        modal_type_embed = self.modal_type_embed_layer(modal_type)

        y = torch.zeros((b,t,256), dtype=audio.dtype, device=audio.device)
        y_mask = torch.zeros((b,1,t), dtype=torch.bool, device=audio.device)

        counts = torch.bincount(modal_type, minlength=3).tolist()
        for i in range(3):
            if counts[i] == 0:
                continue

            idx = torch.where(modal_type == i)

            if i == 0:
                enc, enc_mask, pos = self.kps_encoder(kps[idx], kps_lengths[idx], modal_type_embed[idx])
                y[idx] += enc
                y_mask[idx] += enc_mask

            if i == 1:
                enc, enc_mask, pos = self.audio_encoder(audio[idx], audio_lengths[idx], modal_type_embed[idx])
                y[idx] += enc
                y_mask[idx] += enc_mask

            if i == 2:
                enc, enc_mask, pos = self.audio_encoder(audio[idx], audio_lengths[idx], modal_type_embed[idx])
                y[idx] += enc
                y_mask[idx] += enc_mask

                enc, enc_mask, pos = self.kps_encoder(kps[idx], kps_lengths[idx], modal_type_embed[idx])
                y[idx] += enc

        return y, y_mask.bool()

    def fusion_encode_for_speed(self, audio, audio_lengths, kps, kps_lengths, modal_type):
        time_before_encode = time.time()

        b,t,_ = audio.shape

        assert len(modal_type.shape) == 1

        modal_type_embed = self.modal_type_embed_layer(modal_type)

        y = torch.zeros((b,t,256), dtype=audio.dtype, device=audio.device)
        y_mask = torch.zeros((b,1,t), dtype=torch.bool, device=audio.device)

        counts = torch.bincount(modal_type, minlength=3).tolist()
        for i in range(3):
            if counts[i] == 0:
                continue

            idx = torch.where(modal_type == i)

            if i == 0:
                enc, enc_mask, pos = self.kps_encoder(kps[idx], kps_lengths[idx], modal_type_embed[idx])
                y[idx] += enc
                y_mask[idx] += enc_mask

            if i == 1:
                enc, enc_mask, pos = self.audio_encoder(audio[idx], audio_lengths[idx], modal_type_embed[idx])
                y[idx] += enc
                y_mask[idx] += enc_mask

            if i == 2:
                enc, enc_mask, pos = self.audio_encoder(audio[idx], audio_lengths[idx], modal_type_embed[idx])
                y[idx] += enc
                y_mask[idx] += enc_mask

                enc, enc_mask, pos = self.kps_encoder(kps[idx], kps_lengths[idx], modal_type_embed[idx])
                y[idx] += enc

        time_after_encode = time.time()
        return y, y_mask.bool(), time_after_encode - time_before_encode

    def forward(
        self,
        audio,
        driven,
        init,
        lengths,
        epoch,
        change_epoch,
        target=None,
        emotion=None,
        speaker_kps=None,
        modal_type=None,
        lengths_kps=None,
        encode_type='kps',
    ):
        if target is not None:
            pred = self.predict(
                audio,
                driven,
                init,
                lengths,
                epoch,
                change_epoch,
                target,
                emotion,
                speaker_kps,
                modal_type,
                lengths_kps,
            )
        else:
            pred = self.decode_period_sl_initlast_cat(
                audio,
                driven,
                init,
                lengths,
                change_epoch,
                emotion,
                speaker_kps,
                encode_type,
            )

        if target is not None:
            for i in range(audio.size(0)):
                pred[i, lengths[i] - 1:, :] = 0.
            target_expand = torch.zeros_like(pred)
            target_expand[:, :-1, :] = target
            loss, loss_dict = self.get_loss(
                pred,
                target_expand,
                emotion
            )
            return loss, loss_dict, pred
        return pred

    def visualize(
        self,
        audio,
        driven,
        init,
        lengths,
        epoch,
        change_epoch,
        target=None,
        emotion=None,
        speaker_kps=None,
        modal_type=None,
        lengths_kps=None,
        encode_type='kps',
    ):
        enc_results = {
            'audio': None,
            'visual': None,
            'together': None,
        }

        assert encode_type in ['kps', 'audio', 'together']

        period = 90
        slide = 80
        frame_nums = audio.shape[1]
        current = 0
        while (current + period) < frame_nums:
            audio_cat = audio[:, int(current):int(current + period), :]
            kps_cat = speaker_kps[:, int(current):int(current + period), :]

            lengths = torch.tensor([0]).cuda().long()
            lengths_kps = torch.tensor([period]).cuda().long()
            modal_type = torch.tensor([0]).cuda().long()
            encoder_out, encoder_mask = self.fusion_encode(audio_cat, lengths, kps_cat, lengths_kps, modal_type)
            if current == 0:
                enc_results["visual"] = encoder_out
            else:
                enc_results["visual"] = enc_results["visual"][:, :-(period - slide), :]
                enc_results["visual"] = torch.cat([enc_results["visual"], encoder_out], 1)

            lengths = torch.tensor([period]).cuda().long()
            lengths_kps = torch.tensor([0]).cuda().long()
            modal_type = torch.tensor([1]).cuda().long()
            encoder_out, encoder_mask = self.fusion_encode(audio_cat, lengths, kps_cat, lengths_kps, modal_type)
            if current == 0:
                enc_results["audio"] = encoder_out
            else:
                enc_results["audio"] = enc_results["audio"][:, :-(period - slide), :]
                enc_results["audio"] = torch.cat([enc_results["audio"], encoder_out], 1)

            lengths = torch.tensor([period]).cuda().long()
            lengths_kps = torch.tensor([period]).cuda().long()
            modal_type = torch.tensor([2]).cuda().long()
            encoder_out, encoder_mask = self.fusion_encode(audio_cat, lengths, kps_cat, lengths_kps, modal_type)
            if current == 0:
                enc_results["together"] = encoder_out
            else:
                enc_results["together"] = enc_results["together"][:, :-(period - slide), :]
                enc_results["together"] = torch.cat([enc_results["together"], encoder_out], 1)

            current = current + slide

        return enc_results

    def forward_for_speed_test(
        self,
        audio,
        driven,
        init,
        lengths,
        epoch,
        change_epoch,
        target=None,
        emotion=None,
        speaker_kps=None,
        modal_type=None,
        lengths_kps=None,
        encode_type='kps',
    ):
        if target is not None:
            raise NotImplementedError()

        pred, total_enc_time, total_dec_time = self.decode_period_sl_initlast_cat_for_speed(
            audio,
            driven,
            init,
            lengths,
            change_epoch,
            emotion,
            speaker_kps,
            encode_type,
        )
        return pred, total_enc_time, total_dec_time


@dataclass
class ModelArgs:
    """
    Data class for defining model arguments and hyperparameters.

    Attributes:
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        inter_dim (int): Intermediate dimension for MLP layers.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        n_dense_layers (int): Number of dense layers in the model.
        n_heads (int): Number of attention heads.
        n_routed_experts (int): Number of routed experts for MoE layers.
        n_shared_experts (int): Number of shared experts for MoE layers.
        n_activated_experts (int): Number of activated experts in MoE layers.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited groups for MoE routing.
        score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
        route_scale (float): Scaling factor for routing scores.
        q_lora_rank (int): LoRA rank for query projections.
        kv_lora_rank (int): LoRA rank for key-value projections.
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
        v_head_dim (int): Dimension for value projections.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
        mscale (float): Scaling factor for extended attention.
    """
    dim: int = 256
    moe_inter_dim: int = 256
    # moe
    n_routed_experts: int = 6
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.

    use_bias = False
    # new args
    modal_type_embed_dim: int = 64
    bias_update_rate = 1e-4
    

world_size = 1
rank = 0
class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """
    def __init__(self, args: ModelArgs):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        self.dim = args.dim
        assert args.n_routed_experts % world_size == 0, f"Number of experts must be divisible by world size (world_size={world_size})"
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(args)
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None
                                      for i in range(self.n_routed_experts)])
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: torch.Tensor, modal_type_embed: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x, modal_type_embed)         # modal_type作为条件加入gate

        # print(indices.shape, indices, weights)
        
        y = torch.zeros_like(x)

        #print("input_of_moe:", torch.isnan(x).any())
        for i in range(self.experts_start_idx, self.experts_end_idx):
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            if idx.numel() == 0:
                continue
            #print("expert:", i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
            #print("y", torch.isnan(y).any())


        z = self.shared_experts(x)
        if world_size > 1:
            dist.all_reduce(y)
        res = (y + z).view(shape)
        #print("output_of_moe:", torch.isnan(res).any())
        assert not torch.isnan(res).any()
        return res

def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    return F.linear(x, weight, bias)

class Gate(nn.Module):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

    Attributes:
        dim (int): Dimensionality of input features.
        topk (int): Number of top experts activated for each input.
        n_groups (int): Number of groups for routing.
        topk_groups (int): Number of groups to route inputs to.
        score_func (str): Scoring function ('softmax' or 'sigmoid').
        route_scale (float): Scaling factor for routing weights.
        weight (torch.nn.Parameter): Learnable weights for the gate.
        bias (Optional[torch.nn.Parameter]): Optional bias term for the gate.
    """
    def __init__(self, args: ModelArgs):
        """
        Initializes the Gate module.

        Args:
            args (ModelArgs): Model arguments containing gating parameters.
        """
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.randn(args.n_routed_experts, args.dim + args.modal_type_embed_dim))
        self.bias = nn.Parameter(torch.zeros(args.n_routed_experts), requires_grad = False) if args.use_bias else None      # None

        self.n_routed_experts = args.n_routed_experts
        self.bias_update_rate = args.bias_update_rate

        # self.percentile_target = torch.from_numpy(np.array([0.1,0.1,0.1,0.1,0.3,0.3])).float().cuda()
        self.percentile_target = torch.from_numpy(np.array([0.25, 0.25, 0.25, 0.75, 0.75, 0.75])/3.).float().cuda()


    def forward(self, x: torch.Tensor, modal_type_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.
            modal_type_embed (torch.Tensor): modal_type_embed, influence Score

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """
        #scores = linear(x, self.weight)
        
        bt, _ = x.shape
        b, d = modal_type_embed.shape
        t = bt // b

        # print("in the Gate, modal_type_embed is :", modal_type_embed.shape)
        #print("modal_type_embed has nan", torch.isnan(modal_type_embed).any())
        #print("x in gate has nan", torch.isnan(x).any())
        modal_type_embed = modal_type_embed.unsqueeze(1).repeat(1,t,1).view(bt, d)      #(b,d) -> (b,t,d) -> (bt,d)

        scores_input = torch.cat([x, modal_type_embed], dim = -1)

        # print("scores_input has nan!!!!!!!!!!!", torch.isnan(scores_input).any())
        # print(torch.max(self.weight), torch.max(scores_input))
        # print(torch.min(self.weight), torch.min(scores_input))
        scores = linear(scores_input, self.weight)
        # print("scores has nan!!!!!!!!!!!!", torch.isnan(scores).any())

        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale

        if self.bias is not None and self.bias_update_rate > 0:
            counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts)
            #print("counts:", counts)
            percentile = counts / (bt*self.topk)
            #print("percentile:", percentile)
            percentile_target = self.percentile_target
            #print("percentile_target", percentile_target)
            update_direction = torch.sign(percentile_target.detach() - percentile.detach())
            #print("update_direction", update_direction)
            #print("bias before", self.bias)
            self.bias.data.add_(self.bias_update_rate * update_direction)
            #print("bias after", self.bias)

        return weights.type_as(x), indices

class Expert(nn.Module):
    """
    Expert layer for Mixture-of-Experts (MoE) models.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the Expert layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim)
        self.w2 = nn.Linear(inter_dim, dim)
        self.w3 = nn.Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the MLP layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim)
        self.w2 = nn.Linear(inter_dim, dim)
        self.w3 = nn.Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))



import torch.nn as nn
class MOETransformerEncoderLayer(nn.Module):
    """Encoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
            instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: to use layer_norm after each sub-block.
    """
    def __init__(
        self,
        size: int,
        self_attn: torch.nn.Module,
        feed_forward: torch.nn.Module,
        dropout_rate: float,
        normalize_before: bool = True,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(size, eps=1e-5)
        self.norm2 = nn.LayerNorm(size, eps=1e-5)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),


        moe_condition: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute encoded features.

        Args:
            x (torch.Tensor): (#batch, time, size)
            mask (torch.Tensor): Mask tensor for the input (#batch, time，time),
                (0, 0, 0) means fake mask.
            pos_emb (torch.Tensor): just for interface compatibility
                to ConformerEncoderLayer
            mask_pad (torch.Tensor): does not used in transformer layer,
                just for unified api with conformer.
            att_cache (torch.Tensor): Cache tensor of the KEY & VALUE
                (#batch=1, head, cache_t1, d_k * 2), head * d_k == size.
            cnn_cache (torch.Tensor): Convolution cache in conformer layer
                (#batch=1, size, cache_t2), not used here, it's for interface
                compatibility to ConformerEncoderLayer.
        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time, time).
            torch.Tensor: att_cache tensor,
                (#batch=1, head, cache_t1 + time, d_k * 2).
            torch.Tensor: cnn_cahce tensor (#batch=1, size, cache_t2).

        """
        residual = x
        if self.normalize_before:
            x = self.norm1(x)
        x_att, new_att_cache = self.self_attn(
            x, x, x, mask, cache=att_cache)
        x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = self.feed_forward(x, modal_type_embed=moe_condition)
        x = residual + self.dropout(x)
        if not self.normalize_before:
            x = self.norm2(x)

        fake_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        return x, mask, new_att_cache, fake_cnn_cache

from .utils.mask import add_optional_chunk_mask
class MOETransformerEncoder(BaseEncoder):
    """Transformer encoder module."""
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "abs_pos",
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,

        moe_args = None
    ):
        """ Construct TransformerEncoder

        See Encoder for the meaning of each parameter.
        """
        super().__init__(input_size, output_size, attention_heads,
                         linear_units, num_blocks, dropout_rate,
                         positional_dropout_rate, attention_dropout_rate,
                         input_layer, pos_enc_layer_type, normalize_before,
                         static_chunk_size, use_dynamic_chunk,
                         global_cmvn, use_dynamic_left_chunk)
        self.encoders = torch.nn.ModuleList([
            MOETransformerEncoderLayer(
                output_size,
                MultiHeadedAttention(attention_heads, output_size,
                                     attention_dropout_rate),
                MoE(moe_args), 
                dropout_rate,
                normalize_before) for _ in range(num_blocks)
        ])

    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,

        moe_condition = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed positions in tensor.

        Args:
            xs: padded input tensor (B, T, D)
            xs_lens: input length (B)
            decoding_chunk_size: decoding chunk size for dynamic chunk
                0: default for training, use random dynamic chunk.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
            num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
                >=0: use num_decoding_left_chunks
                <0: use all left chunks
        Returns:
            encoder output tensor xs, and subsampled masks
            xs: padded output tensor (B, T' ~= T/subsample_rate, D)
            masks: torch.Tensor batch padding mask after subsample
                (B, 1, T' ~= T/subsample_rate)
        """
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, masks = self.embed(xs, masks)
        mask_pad = masks  # (B, 1, T/subsample_rate)
        #print('mask_pad',mask_pad.device)
        chunk_masks = add_optional_chunk_mask(xs, masks,
                                              self.use_dynamic_chunk,
                                              self.use_dynamic_left_chunk,
                                              decoding_chunk_size,
                                              self.static_chunk_size,
                                              num_decoding_left_chunks)
        #print('chunk_masks', chunk_masks.device)
        for layer in self.encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad, moe_condition=moe_condition)
        if self.normalize_before:
            xs = self.after_norm(xs)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return xs, masks, pos_emb

if __name__ == "__main__":
    # torch.set_default_dtype(torch.bfloat16)
    # torch.set_default_device("cuda")
    # torch.manual_seed(0)
    # args = ModelArgs()
    # x = torch.randint(0, args.vocab_size, (2, 128))
    # model = Transformer(args)
    # print(model(x).size())



    args = ModelArgs()
    x = torch.randn((4,90,256))
    model = MoE(args)
    print(model(x).size())
