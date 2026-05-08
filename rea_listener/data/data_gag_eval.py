from bisect import bisect_right
from math import ceil
import numpy as np
import os
import os.path as osp
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.distributed as dist
from copy import deepcopy
from box import Box
import torch.nn.utils.rnn as rnn_utils
from collections import Counter

import pickle

from rea_listener.paths import build_vico_layout

FEATURE_DIR = 'features'
METADATA_DIR = 'metadata'
AUDIO_FEATS_DIR = 'audio_features'
VIDEO_FEATS_DIR_SPE = 'speaker_coef_no_basetm'
VIDEO_FEATS_DIR_SPE_KEYPOINT = 'video_keypoint_features'
VIDEO_FEATS_DIR_LIS = 'listener_coef_no_basetm'
class VicoDataset(data.Dataset):
    def __init__(self, root, task, dynam_3dmm_transform=None, audio_transform=None, kp_transform = None, keypoint = 104, mode="test",
                 pred_emotion_path = None):
        self.root = root
        self.layout = build_vico_layout(root)
        assert task in ['listener', 'speaker']
        self.task = task
        #self.video_root = osp.join(self.root, FEATURE_DIR, VIDEO_FEATS_DIR)
        #self.audio_root = osp.join(self.root, FEATURE_DIR, AUDIO_FEATS_DIR)
        self.video_root = self.root
        self.audio_root = osp.join(self.root, AUDIO_FEATS_DIR)

        # anno_file = osp.join(self.root, METADATA_DIR, 'data.csv')
        # self.anno_df = pd.read_csv(anno_file)
        # self.anno_df = self.anno_df.reset_index()

        assert mode in ["test", "ood"]

        self.anno_df = pd.read_excel(self.layout.metadata_path)
        # self.anno_df = self.anno_df[self.anno_df["data_split"] == mode]
        self.anno_df = self.anno_df[self.anno_df["data_split"].isin(["test", "ood"])]

        assert len(self.anno_df) > 0
        self.anno_df = self.anno_df.reset_index()


        with open(self.layout.coef_mean_path, 'rb') as file:
            self.norm_mean = pickle.load(file) 
        with open(self.layout.coef_std_path, 'rb') as file:
            self.norm_std = pickle.load(file) 
        self.coef_keys = ['expcode', 'posecode', 'eyecode', 'transform_matrix_angles', 'transform_matrix_trans']
        
        

        raw_data = []
        for row_idx, row in self.anno_df.iterrows():
            audio = np.load(f'{self.audio_root}/{row.audio}.npy')
            #speaker = torch.load(f'{self.video_root}/{row.audio}.speaker.bin')
            #listener = torch.load(f'{self.video_root}/{row.audio}.listener.bin')
            
            speaker = f'{self.video_root}/{VIDEO_FEATS_DIR_SPE}/{row.speaker}/smoothed.pkl'         # load speaker motion
            speaker = torch.from_numpy(self.load_coef_and_norm(speaker)) 
            listener = f'{self.video_root}/{VIDEO_FEATS_DIR_LIS}/{row.listener}/smoothed.pkl'       # load listener motion
            listener = torch.from_numpy(self.load_coef_and_norm(listener))
            #print(speaker.size(), listener.size())

            if keypoint == 104:
                speaker_kp = f'{self.video_root}/{VIDEO_FEATS_DIR_SPE_KEYPOINT}/{row.speaker}_104.npy'
            else:
                speaker_kp = f'{self.video_root}/{VIDEO_FEATS_DIR_SPE_KEYPOINT}/{row.speaker}_468.npy'

            speaker_kp = np.load(speaker_kp)
            # if row_idx == 0:
            #     print(row)
            #     print(speaker_kp)

            ##################
            # 测试，使用固定的kps时，结果如何
            #speaker_kp_temp = np.load('/path/to/vico/template_keypoint_refine.npy')
            #speaker_kp = np.repeat(np.expand_dims(speaker_kp_temp, axis=0), speaker_kp.shape[0], axis=0)
            ##################

            speaker_kp = torch.from_numpy(speaker_kp)
            speaker_kp = kp_transform(speaker_kp).view(speaker_kp.shape[0], -1)     # t, 208  /  t, 936


            if row.attitude == 'positive':
                attitude = torch.tensor(0)
            elif row.attitude == 'neutral':
                attitude = torch.tensor(1)
            else:
                attitude = torch.tensor(2)

            emotion_list = []
            emotion_map = {'Anger': 0, 'Disgust': 0, 'Fear': 5, 'Happiness': 3, 'Neutral': 4, 'Sadness': 5, 'Surprise': 3}
                
            if pred_emotion_path is None:
                emo_anno_path = self.layout.emotion_dir / f"{row['listener']}.xlsx"
                emotion_data = pd.read_excel(emo_anno_path)
                emotion_column = emotion_data["facial_expression"].tolist()
                emotion_column = [emotion_map[x] for x in emotion_column]
                
                nframes = listener.shape[0]
                if nframes > 90:
                    for frame_start_idx in range(0, nframes, 80):
                        if frame_start_idx + 90 > nframes:
                            break
                        counter = Counter(emotion_column[frame_start_idx: frame_start_idx + 90])
                        emotion_list.append(torch.LongTensor([counter.most_common(1)[0][0]]))
                    counter = Counter(emotion_column[-90:])
                    emotion_list.append(torch.LongTensor([counter.most_common(1)[0][0]]))
                else:
                    counter = Counter(emotion_column)
                    emotion_list.append(torch.LongTensor([counter.most_common(1)[0][0]]))


            else:
                emo_anno_path = os.path.join(pred_emotion_path, row["listener"]+'_fine.xlsx')
                emotion_data = pd.read_excel(emo_anno_path)
                dict_result = dict(zip(emotion_data['frame'], emotion_data['emotion']))

                nframes = listener.shape[0]
                if nframes > 90:
                    for frame_start_idx in range(0, nframes, 80):
                        if frame_start_idx + 90 > nframes:
                            break
                        emotion_list.append(torch.LongTensor([emotion_map[dict_result[frame_start_idx]]]))
                    emotion_list.append(torch.LongTensor([emotion_map[dict_result[nframes-90]]]))
                else:
                    emotion_list.append(torch.LongTensor([emotion_map[dict_result[0]]]))

            #print("empty audio")
            attitude = F.one_hot(attitude, num_classes=3)
            raw_data.append({
                #'audio': np.zeros_like(audio),
                'audio': audio,
                'speaker': speaker,
                'listener': listener,
                'attitude': attitude,
                'emotion': emotion_list,
                'speaker_kp': speaker_kp
            })
        self.data = raw_data

        self.dynam_3dmm_transform = dynam_3dmm_transform
        self.audio_transform = audio_transform
        self.kp_transform = kp_transform

    def load_coef_and_norm(self, path):
        with open(path, "rb") as file:
            data = pickle.load(file)
        num_frames = data["num_frames"]
        #if num_frames + 3 != len(data.keys()):
        #print(path, num_frames)
        #print(data.keys())
        new_dict = {}
        for key in self.coef_keys:
            coef = [data[str(frame)][key] for frame in range(num_frames)]
            coef = np.array(coef)
            if key == 'posecode':
                coef = coef[:,3:]
                new_dict[key] = (coef - self.norm_mean[key][3:]) / (self.norm_std[key][3:])
            else:
                new_dict[key] = (coef - self.norm_mean[key]) / (self.norm_std[key])

        return np.concatenate(
            [new_dict["expcode"],
             new_dict["posecode"],
             new_dict["eyecode"],
             new_dict["transform_matrix_angles"],
             new_dict["transform_matrix_trans"]],
            axis = -1
        )



    def __getitem__(self, idx):
        attitude = self.data[idx]['attitude']
        row = self.anno_df.iloc[idx]

        #frame_indexs = self.data[idx]['frame_indexs']
        audio = torch.from_numpy(self.data[idx]['audio'])#[frame_indexs]
        speaker_video = self.data[idx]['speaker']
        listener_video = self.data[idx]['listener']
        emotion = self.data[idx]['emotion']
        speaker_kp = self.data[idx]['speaker_kp']
        #listener_3dmm_fixed, listener_3dmm_dynam = self.load_3dmm(listener_video, frame_indexs)
        #speaker_3dmm_fixed,  speaker_3dmm_dynam  = self.load_3dmm(speaker_video, frame_indexs)

        speaker_3dmm_dynam = speaker_video.float()
        listener_3dmm_dynam = listener_video.float()
        speaker_kp_dynam = speaker_kp.contiguous().float()
        
        if self.task == 'listener':
            driven_3dmm_dynam = speaker_3dmm_dynam
            init_3dmm_dynam, target_3dmm_dynam = torch.split(listener_3dmm_dynam, [1, listener_3dmm_dynam.size(0) - 1])
        else:
            raise NotImplementedError("")
            driven_3dmm_dynam = listener_3dmm_dynam
            init_3dmm_dynam, target_3dmm_dynam = torch.split(speaker_3dmm_dynam, [1, speaker_3dmm_dynam.size(0) - 1])
        
        if self.audio_transform is not None:
            audio = self.audio_transform(audio)

        if self.dynam_3dmm_transform is not None:
            driven_3dmm_dynam = self.dynam_3dmm_transform(driven_3dmm_dynam)
            init_3dmm_dynam = self.dynam_3dmm_transform(init_3dmm_dynam)
            target_3dmm_dynam = self.dynam_3dmm_transform(target_3dmm_dynam)

        driven_signal = driven_3dmm_dynam
        init_signal = init_3dmm_dynam
        target_signal = target_3dmm_dynam

        return audio, driven_signal, init_signal, target_signal, row, attitude, emotion, speaker_kp_dynam

    def load_3dmm(self, data, frame_indexs):
        id_gamma_tex    = torch.from_numpy(data['id.gamma.tex'][frame_indexs, :]).float()
        angle_exp_trans = torch.from_numpy(data['angle.exp.trans'][frame_indexs, :]).float()
        return id_gamma_tex, angle_exp_trans

    def __len__(self):
        return len(self.data)

def get_dataset(config, task, keypoint = 104, **kwargs):
    #mfcc_mean_std_info = torch.load(osp.join(config.root, FEATURE_DIR, 'audio_feats_mean_std.bin'))
    #mfcc_mean, mfcc_std = mfcc_mean_std_info['mean'], mfcc_mean_std_info['std']
    #audio_transform = transforms.Lambda(lambda e: (e - mfcc_mean) / mfcc_std)

    #dynam_mean_std_info = torch.load(osp.join(config.root, FEATURE_DIR, 'video_feats_mean_std.bin'))['angle.exp.trans']
    #dynam_mean, dynam_std = dynam_mean_std_info['mean'], dynam_mean_std_info['std']
    #dynam_3dmm_transform = transforms.Lambda(lambda e: (e - dynam_mean) / dynam_std)

    layout = build_vico_layout(config['root'])
    mean_std = torch.load(layout.audio_stats_path)
    audio_mean = mean_std['mean'].unsqueeze(0).numpy()
    audio_std = mean_std['std'].unsqueeze(0).numpy()
    audio_transform = transforms.Lambda(lambda e: (e - audio_mean) / audio_std)

    assert keypoint in [104, 468]
    if keypoint == 104:
        mean_std_kp = torch.load(layout.keypoint_stats_104_path)
    else:
        mean_std_kp = torch.load(layout.keypoint_stats_468_path)
    kp_mean = mean_std_kp['mean'].unsqueeze(0).numpy()
    kp_std = mean_std_kp['std'].unsqueeze(0).numpy()
    kp_transform = transforms.Lambda(lambda e: (e - kp_mean) / kp_std)



    dataset = VicoDataset(
        config['root'],
        task,
        dynam_3dmm_transform=None,
        #dynam_3dmm_transform=dynam_3dmm_transform,
        audio_transform=audio_transform,
        kp_transform = kp_transform,
        keypoint = keypoint,
        **kwargs
    )
    return dataset

def collate_fn(batch):
    audio = [e[0] for e in batch]
    driven = [e[1] for e in batch]
    init = [e[2] for e in batch]
    target = [e[3] for e in batch]
    rows = [e[4] for e in batch]
    attitude = [e[5] for e in batch]
    speaker_kp_dynam = [e[7] for e in batch]
    lengths = torch.from_numpy(np.array([e.size(0) for e in driven]))

    audio = rnn_utils.pad_sequence(audio, batch_first=True)
    driven = rnn_utils.pad_sequence(driven, batch_first=True)
    target = rnn_utils.pad_sequence(target, batch_first=True)
    init = torch.vstack(init)
    attitude = torch.vstack(attitude)
    speaker_kp_dynam = rnn_utils.pad_sequence(speaker_kp_dynam, batch_first=True)
    
    
    emotion = [e[6] for e in batch]
    emotion = emotion[0]
    #print("audio:", audio.shape)
    #print("emotion:", len(emotion))
    emotion = [e.unsqueeze(0) for e in emotion]

    return audio, driven, init, target, lengths, rows, attitude, emotion, speaker_kp_dynam   # a list of LongTensor (1, 1)

def get_data_loader(config, task, pred_emotion_path = None, keypoint = 104):
    dataset = get_dataset(config, task, pred_emotion_path = pred_emotion_path, keypoint = keypoint)
    loader = DataLoader(
        dataset=dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=config['batch_size'],
        drop_last=False,
        num_workers=config['num_workers'],
        pin_memory=True,
    )
    return loader



if __name__ == '__main__':
    # config = {
    #     'batch_size': 4,
    #     'num_workers': 1,

    #     'root': 
    # }
    # ds = VicoDataset('/path/to/vico',
    #                 'listener',
    #                 time_size = 90,
    #                 mode = "train")
    
    ds = get_dataset(config = {'root':'/path/to/vico'},
                     task = 'listener',
                     mode = "train")
    
    d0 = ds[0]

    for what in d0:
        print(type(what))
        if isinstance(what, torch.Tensor) or isinstance(what, np.ndarray):
            print(what.shape)
    
