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
from tqdm import tqdm
from collections import Counter
import pickle
import random

from rea_listener.paths import build_vico_layout

FEATURE_DIR = 'features'
METADATA_DIR = 'metadata'
AUDIO_FEATS_DIR = 'audio_features'
VIDEO_FEATS_DIR_SPE = 'speaker_coef_no_basetm'
VIDEO_FEATS_DIR_SPE_KEYPOINT = 'video_keypoint_features'
VIDEO_FEATS_DIR_LIS = 'listener_coef_no_basetm'
class VicoDataset(data.Dataset):
    def __init__(self, root, task, time_size, dynam_3dmm_transform=None, audio_transform=None, kp_transform = None, keypoint = 104, mode="train", random_mask = -1):
        self.root = root
        self.layout = build_vico_layout(root)
        assert task in ['listener', 'speaker']
        self.task = task

        self.video_root = self.root
        self.audio_root = osp.join(self.root, AUDIO_FEATS_DIR)



        if mode == "train":
            mode = ["train", "val"]
        else:
            mode = [mode]

        self.anno_df = pd.read_excel(self.layout.metadata_path)
        self.anno_df = self.anno_df[self.anno_df["data_split"].isin(mode)]
        assert len(self.anno_df) > 0
        self.anno_df = self.anno_df.reset_index()


        with open(self.layout.coef_mean_path, 'rb') as file:
            self.norm_mean = pickle.load(file) 
        with open(self.layout.coef_std_path, 'rb') as file:
            self.norm_std = pickle.load(file) 
        self.coef_keys = ['expcode', 'posecode', 'eyecode', 'transform_matrix_angles', 'transform_matrix_trans']

        raw_data = []
        for row_idx, row in tqdm(self.anno_df.iterrows()):
            
            # if row_idx >=50:
            #     break

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
            speaker_kp = torch.from_numpy(speaker_kp)
            speaker_kp = kp_transform(speaker_kp).view(speaker_kp.shape[0], -1)     # t, 208  /  t, 936



            emo_anno_path = self.layout.emotion_dir / f"{row['listener']}.xlsx"
            emotion_data = pd.read_excel(emo_anno_path)
            emotion_column = emotion_data["facial_expression"].tolist()
            emotion_map = {'Anger': 0, 'Disgust': 0, 'Fear': 5, 'Happiness': 3, 'Neutral': 4, 'Sadness': 5, 'Surprise': 3}
            emotion_column = [emotion_map[x] for x in emotion_column]

            if row.attitude == 'positive':
                attitude = torch.tensor(0)
            elif row.attitude == 'neutral':
                attitude = torch.tensor(1)
            else:
                attitude = torch.tensor(2)

            attitude = F.one_hot(attitude, num_classes=3)

            nframes = len(audio)
            if nframes > time_size:
                for i in range(nframes - time_size // 3 + 1):
                    counter = Counter(emotion_column[i: min(i + time_size, nframes)])
                    raw_data.append({
                        'audio': audio,
                        'speaker': speaker,
                        'listener': listener,
                        'frame_indexs': list(range(i, min(i + time_size, nframes))),
                        'row_idx': row_idx,
                        'attitude': attitude,
                        'emotion' : torch.LongTensor([counter.most_common(1)[0][0]]),
                        'speaker_kp': speaker_kp
                    })
            else:
                #print()
                counter = Counter(emotion_column)
                raw_data.append({
                    'audio': audio,
                    'speaker': speaker,
                    'listener': listener,
                    'frame_indexs': list(range(nframes)),
                    'row_idx': row_idx,
                    'attitude': attitude,
                    'emotion' : torch.LongTensor([counter.most_common(1)[0][0]]),
                    'speaker_kp': speaker_kp
                })
        self.data = raw_data

        self.dynam_3dmm_transform = dynam_3dmm_transform
        self.audio_transform = audio_transform
        self.kp_transform = kp_transform

        assert random_mask < 0.5
        self.random_mask = random_mask

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
        row_idx = self.data[idx]['row_idx']
        attitude = self.data[idx]['attitude']
        row = self.anno_df.iloc[row_idx]

        frame_indexs = self.data[idx]['frame_indexs']
        audio = torch.from_numpy(self.data[idx]['audio'])[frame_indexs]
        speaker_video = self.data[idx]['speaker']
        listener_video = self.data[idx]['listener']

        emotion = self.data[idx]['emotion']

        speaker_kp = self.data[idx]['speaker_kp']

        speaker_3dmm_dynam = speaker_video[frame_indexs,:].contiguous().float()
        listener_3dmm_dynam = listener_video[frame_indexs,:].contiguous().float()
        speaker_kp_dynam = speaker_kp[frame_indexs,:].contiguous().float()
        
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

        if self.random_mask > 0:
            # mask audio
            route = random.random()
            if route < self.random_mask:
                audio = torch.zeros_like(audio)
                modal_type = torch.LongTensor([0])
                audio_length = 0
                kps_length = speaker_kp_dynam.shape[0]
            
            # mask kps
            elif route < 2 * self.random_mask:
                speaker_kp_dynam = torch.zeros_like(speaker_kp_dynam)
                modal_type = torch.LongTensor([1])
                kps_length = 0
                audio_length = audio.shape[0]

            else:
                modal_type = torch.LongTensor([2])
                audio_length = audio.shape[0]
                kps_length = speaker_kp_dynam.shape[0]


            return audio, driven_signal, init_signal, target_signal, row, attitude, emotion, speaker_kp_dynam, \
                modal_type, audio_length, kps_length

        return audio, driven_signal, init_signal, target_signal, row, attitude, emotion, speaker_kp_dynam

    def load_3dmm(self, data, frame_indexs):
        id_gamma_tex    = torch.from_numpy(data['id.gamma.tex'][frame_indexs, :]).float()
        angle_exp_trans = torch.from_numpy(data['angle.exp.trans'][frame_indexs, :]).float()
        return id_gamma_tex, angle_exp_trans

    def __len__(self):
        return len(self.data)

def get_dataset(config, task, time_size, keypoint = 104, **kwargs):
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
        time_size,
        dynam_3dmm_transform=None,
        #dynam_3dmm_transform=dynam_3dmm_transform,
        audio_transform=audio_transform,
        kp_transform = kp_transform,
        keypoint = keypoint,
        **kwargs
    )
    return dataset

def collate_fn_random_mask(batch):
    audio = [e[0] for e in batch]
    driven = [e[1] for e in batch]
    init = [e[2] for e in batch]
    target = [e[3] for e in batch]
    rows = [e[4] for e in batch]
    attitude = [e[5] for e in batch]
    emotion = [e[6] for e in batch]
    speaker_kp_dynam = [e[7] for e in batch]
    # lengths = torch.from_numpy(np.array([e.size(0) for e in driven]))


    modal_type = [e[8] for e in batch]
    lengths = torch.from_numpy(np.array([e[9] for e in batch]))
    lengths_kps = torch.from_numpy(np.array([e[10] for e in batch]))

    audio = rnn_utils.pad_sequence(audio, batch_first=True)
    driven = rnn_utils.pad_sequence(driven, batch_first=True)
    target = rnn_utils.pad_sequence(target, batch_first=True)
    init = torch.vstack(init)
    attitude = torch.vstack(attitude)
    emotion = torch.vstack(emotion)

    modal_type = torch.vstack(modal_type)

    speaker_kp_dynam = rnn_utils.pad_sequence(speaker_kp_dynam, batch_first=True)
    #print(emotion.shape)
    #exit()
    return audio, driven, init, target, lengths, rows, attitude, emotion, speaker_kp_dynam, modal_type, lengths_kps

def collate_fn(batch):
    audio = [e[0] for e in batch]
    driven = [e[1] for e in batch]
    init = [e[2] for e in batch]
    target = [e[3] for e in batch]
    rows = [e[4] for e in batch]
    attitude = [e[5] for e in batch]
    emotion = [e[6] for e in batch]
    speaker_kp_dynam = [e[7] for e in batch]

    lengths = torch.from_numpy(np.array([e.size(0) for e in driven]))

    audio = rnn_utils.pad_sequence(audio, batch_first=True)
    driven = rnn_utils.pad_sequence(driven, batch_first=True)
    target = rnn_utils.pad_sequence(target, batch_first=True)
    init = torch.vstack(init)
    attitude = torch.vstack(attitude)
    emotion = torch.vstack(emotion)
    speaker_kp_dynam = rnn_utils.pad_sequence(speaker_kp_dynam, batch_first=True)
    #print(emotion.shape)
    #exit()
    return audio, driven, init, target, lengths, rows, attitude, emotion, speaker_kp_dynam

def get_data_loader(config, task, time_size, keypoint = 104, random_mask = -1):
    dataset = get_dataset(config, task, time_size, keypoint, random_mask = random_mask)
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler = data.DistributedSampler(
        dataset,
        num_replicas=num_tasks,
        rank=global_rank,
        shuffle=True,
    )
    loader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        collate_fn=collate_fn if random_mask <=0 else collate_fn_random_mask,
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
                     time_size = 90,
                     mode = "train")
    
    d0 = ds[0]

    for what in d0:
        print(type(what))
        if isinstance(what, torch.Tensor) or isinstance(what, np.ndarray):
            print(what.shape)
    
    print(d0[-1])
    
