import math
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.data import crop_scale

def coco2h36m(x):
    '''
        Input: x (T x V x C)
        COCO:                   H36M:
        0: nose,                0: root,
        1: left_eye,            1: right_hip,
        2: right_eye,           2: right_knee,
        3: left_ear,            3: right_ankle,
        4: right_ear,           4: left_hip,
        5: left_shoulder,       5: left_knee,
        6: right_shoulder,      6: left_ankle,
        7: left_elbow,          7: belly,
        8: right_elbow,         8: neck,
        9: left_wrist,          9: nose,
        10: right_wrist,        10: head,
        11: left_hip,           11: left_shoulder,
        12: right_hip,          12: left_elbow,
        13: left_knee,          13: left_wrist,
        14: right_knee,         14: right_shoulder,
        15: left_ankle,         15: right_elbow,
        16: right_ankle         16: right_wrist
    '''
    y = np.zeros(x.shape)
    y[:, 0, :] = (x[:, 11, :] + x[:, 12, :]) * 0.5
    y[:, 1, :] = x[:, 12, :]
    y[:, 2, :] = x[:, 14, :]
    y[:, 3, :] = x[:, 16, :]
    y[:, 4, :] = x[:, 11, :]
    y[:, 5, :] = x[:, 13, :]
    y[:, 6, :] = x[:, 15, :]
    y[:, 8, :] = (x[:, 5, :] + x[:, 6, :]) * 0.5
    y[:, 7, :] = (y[:, 0, :] + y[:, 8, :]) * 0.5
    y[:, 9, :] = x[:, 0, :]
    y[:, 10, :] = (x[:, 1, :] + x[:, 2, :]) * 0.5
    y[:, 11, :] = x[:, 5, :]
    y[:, 12, :] = x[:, 7, :]
    y[:, 13, :] = x[:, 9, :]
    y[:, 14, :] = x[:, 6, :]
    y[:, 15, :] = x[:, 8, :]
    y[:, 16, :] = x[:, 10, :]
    
    return y

def read_input(json_path, vid_size, scale_range, focus):
    with open(json_path, 'r') as read_file:
        results = json.load(read_file)
    
    keypoints_all = []
    for item in results:
        if focus != None and item['idx'] != focus: # 어차피 실행 안됨
            continue
        keypoints = np.array(item['keypoints']).reshape([-1, 3])
        keypoints_all.append(keypoints)
    keypoints_all = np.array(keypoints_all)
    keypoints_all = coco2h36m(keypoints_all)
    
    if vid_size:
        w, h = vid_size
        scale = min(w, h) / 2.0
        # keypoints_all[:, :, :2]: 좌표에 해당하는 x, y만 읽어와서 처리; score는 처리 X
        keypoints_all[:, :, :2] = keypoints_all[:, :, :2] - np.array([w, h]) / 2.0
        keypoints_all[:, :, :2] = keypoints_all[:, :, :2] / scale
        motion = keypoints_all
    if scale_range:
        motion = crop_scale(keypoints_all, scale_range)
    
    return motion.astype(np.float32)

class DetDataset(Dataset):
    def __init__(self, json_path, clip_len, vid_size=None, scale_range=None, focus=None):
        self.json_path = json_path
        self.clip_len = clip_len
        self.vid_all = read_input(json_path, vid_size, scale_range, focus)
    
    def __len__(self):
        return math.ceil(len(self.vid_all) / self.clip_len)
    
    def __getitem__(self, index):
        st = index * self.clip_len
        end = min((index + 1) * self.clip_len, len(self.vid_all))
        
        return self.vid_all[st: end]