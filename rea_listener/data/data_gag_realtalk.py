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
import json

from scipy.spatial.transform import Rotation as R
from rea_listener.paths import build_realtalk_layout

def rotation_matrix_to_angles(R_matrix):
    r = R.from_matrix(R_matrix)
    return r.as_euler('zyx', degrees=False)

AUDIO_FEATS_DIR = 'audio_features'
VIDEO_FEATS_DIR_SPE = 'track_realtalk_speaker'
VIDEO_FEATS_DIR_SPE_KEYPOINT = 'video_keypoint_features'
VIDEO_FEATS_DIR_LIS = 'track_realtalk_listener'
class RealTalkDataset(data.Dataset):
    def __init__(self, root, task, time_size, dynam_3dmm_transform=None, audio_transform=None, kp_transform = None, keypoint = 104, mode="train", random_mask = -1):
        self.root = root
        self.layout = build_realtalk_layout(root)
        assert task in ['listener', 'speaker']
        self.task = task

        self.video_root = self.root
        self.audio_root = osp.join(self.root, AUDIO_FEATS_DIR)



        assert mode in ["train", "test"]

        with open(self.layout.split_path, 'r') as f:
            self.anno_df = json.load(f)
        
        self.anno_df = self.anno_df[mode]

        with open(self.layout.coef_norm_path, 'rb') as f:
            norm_mean_std = pickle.load(f)

        self.norm_mean = norm_mean_std["mean"]
        self.norm_std = norm_mean_std["std"]
        self.coef_keys = ['expcode', 'posecode', 'eyecode', 'transform_matrix_angles', 'transform_matrix_trans']

        raw_data = []
        for row_idx, row in tqdm(enumerate(self.anno_df)):
            
            # if row_idx >=50:
            #     break

            filename = "_".join(map(str, row["id"]))

            audio = np.load(f'{self.audio_root}/{filename}.npy')
            #speaker = torch.load(f'{self.video_root}/{row.audio}.speaker.bin')
            #listener = torch.load(f'{self.video_root}/{row.audio}.listener.bin')
            
            speaker = f'{self.video_root}/{VIDEO_FEATS_DIR_SPE}/{filename}.mp4/smoothed.pkl'         # load speaker motion
            speaker = torch.from_numpy(self.load_coef_and_norm(speaker, filename)) 
            listener = f'{self.video_root}/{VIDEO_FEATS_DIR_LIS}/{filename}.mp4/smoothed.pkl'       # load listener motion
            listener = torch.from_numpy(self.load_coef_and_norm(listener, filename))
            # print("size:", speaker.size(), listener.size())

            if keypoint == 104:
                speaker_kp = f'{self.video_root}/{VIDEO_FEATS_DIR_SPE_KEYPOINT}/{filename}_104.npy'
            else:
                speaker_kp = f'{self.video_root}/{VIDEO_FEATS_DIR_SPE_KEYPOINT}/{filename}_468.npy'

            speaker_kp = np.load(speaker_kp)
            speaker_kp = torch.from_numpy(speaker_kp)
            speaker_kp = kp_transform(speaker_kp).view(speaker_kp.shape[0], -1)     # t, 208  /  t, 936



            emo_anno_path = self.layout.emotion_dir / f"{filename}.csv"
            emotion_data = pd.read_csv(emo_anno_path)
            emotion_column = emotion_data["facial_expression"].tolist()
            emotion_map = {'Anger': 0, 'Disgust': 0, 'Fear': 5, 'Happiness': 3, 'Neutral': 4, 'Sadness': 5, 'Surprise': 3}
            emotion_column = [emotion_map[x] for x in emotion_column]

            nframes = len(audio)
            if nframes > time_size:
                for i in range(0, nframes - time_size // 3 + 1, 2):
                    if i>=nframes:
                        continue
                    counter = Counter(emotion_column[i: min(i + time_size, nframes)])
                    raw_data.append({
                        'audio': audio,
                        'speaker': speaker,
                        'listener': listener,
                        'frame_indexs': list(range(i, min(i + time_size, nframes))),
                        'row_idx': row_idx,
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
                    'emotion' : torch.LongTensor([counter.most_common(1)[0][0]]),
                    'speaker_kp': speaker_kp
                })
        self.data = raw_data

        self.dynam_3dmm_transform = dynam_3dmm_transform
        self.audio_transform = audio_transform
        self.kp_transform = kp_transform

        assert random_mask < 0.5
        self.random_mask = random_mask

    def load_coef_and_norm(self, path, name):
        with open(path, "rb") as file:
            data = pickle.load(file)
        num_frames = 384

        new_dict = {}
        for key in self.coef_keys:
            if key == 'transform_matrix_angles':
                coef = [rotation_matrix_to_angles(data[name + '_' + str(frame)]['transform_matrix'][:, :3]) for frame in range(num_frames)]
            elif key == 'transform_matrix_trans':
                coef = [data[name + '_' + str(frame)]['transform_matrix'][:, 3] for frame in range(num_frames)]
            else:
                coef = [data[name + '_' + str(frame)][key] for frame in range(num_frames)]

            coef = np.array(coef)

            if key == 'transform_matrix_angles':
                coef[:,0][coef[:,0]<0] += 2*np.pi       # 三个旋转角度中的0和2，分布集中在-pi和pi周围（范围的端点），平移到一起（映射到0-2*pi，集中在pi附近）
                coef[:,2][coef[:,2]<0] += 2*np.pi

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
        row = self.anno_df[row_idx]

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
       
        if self.audio_transform is not None:
            audio = self.audio_transform(audio)

        # if self.dynam_3dmm_transform is not None:
        #     driven_3dmm_dynam = self.dynam_3dmm_transform(driven_3dmm_dynam)
        #     init_3dmm_dynam = self.dynam_3dmm_transform(init_3dmm_dynam)
        #     target_3dmm_dynam = self.dynam_3dmm_transform(target_3dmm_dynam)

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


            return audio, driven_signal, init_signal, target_signal, row, None, emotion, speaker_kp_dynam, \
                modal_type, audio_length, kps_length

        return audio, driven_signal, init_signal, target_signal, row, None, emotion, speaker_kp_dynam

    def load_3dmm(self, data, frame_indexs):
        id_gamma_tex    = torch.from_numpy(data['id.gamma.tex'][frame_indexs, :]).float()
        angle_exp_trans = torch.from_numpy(data['angle.exp.trans'][frame_indexs, :]).float()
        return id_gamma_tex, angle_exp_trans

    def __len__(self):
        return len(self.data)

def get_dataset(config, task, time_size, keypoint = 104, **kwargs):
    layout = build_realtalk_layout(config['root'])
    mean_std = torch.load(layout.audio_stats_path)
    audio_mean = mean_std['mean'].unsqueeze(0).numpy()
    audio_std = mean_std['std'].unsqueeze(0).numpy()
    audio_transform = transforms.Lambda(lambda e: (e - audio_mean) / audio_std)

    assert keypoint in [104, 468]
    if keypoint == 104:
        mean_std_kp = torch.load(layout.keypoint_stats_104_path)
    else:
        raise NotImplementedError("")
    
    kp_mean = mean_std_kp['mean'].unsqueeze(0).numpy()
    kp_std = mean_std_kp['std'].unsqueeze(0).numpy()
    kp_transform = transforms.Lambda(lambda e: (e - kp_mean) / kp_std)

    dataset = RealTalkDataset(
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
    attitude = None
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
    attitude = None
    emotion = [e[6] for e in batch]
    speaker_kp_dynam = [e[7] for e in batch]

    lengths = torch.from_numpy(np.array([e.size(0) for e in driven]))

    audio = rnn_utils.pad_sequence(audio, batch_first=True)
    driven = rnn_utils.pad_sequence(driven, batch_first=True)
    target = rnn_utils.pad_sequence(target, batch_first=True)
    init = torch.vstack(init)
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


def get_data_loader_no_dist(config, task, time_size, keypoint = 104, random_mask = -1):
    dataset = get_dataset(config, task, time_size, keypoint, random_mask = random_mask)
    loader = DataLoader(
        dataset=dataset,
        collate_fn=collate_fn if random_mask <=0 else collate_fn_random_mask,
        batch_size=config['batch_size'],
        drop_last=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        shuffle=True
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
    
    ds = get_dataset(config = {'root':'/path/to/realtalk'},
                     task = 'listener',
                     time_size = 90,
                     mode = "train",
                     random_mask = 0.2)
    
    d0 = ds[0]

    for what in d0:
        print(type(what))
        if isinstance(what, torch.Tensor) or isinstance(what, np.ndarray):
            print(what.shape)

    print(d0[4])
    
    print(len(ds))
    
