import os
import imageio
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.tools import *
from utils.learning import load_backbone
from utils.data import flip_data
from data.dataset_coco import DetDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='./configs/MB_ft_h36m_global_lite.yaml', help='Path to the config file.')
    parser.add_argument('-e', '--evaluate', default='./checkpoint/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin')
    parser.add_argument('-j', '--json_path', type=str, help='Detected 2D pose json path')
    parser.add_argument('-v', '--vid_path', type=str, help='video path')
    parser.add_argument('-o', '--out_path', type=str, help='output path')
    parser.add_argument('--pixel', action='store_true', help='align with pixel coordinates')
    parser.add_argument('--focus', type=int, default=None, help='target person id')
    parser.add_argument('--clip_len', type=int, default=243, help='clip length for network input')
    opts = parser.parse_args()
    
    return opts

opts = parse_args()
args = get_config(opts.config)

model_backbone = load_backbone(args)
if torch.cuda.is_available():
    model_backbone = nn.DataParallel(model_backbone)
    model_backbone = model_backbone.cuda()

print('Loading checkpoint', opts.evaluate)
checkpoint = torch.load(opts.evaluate, map_location=lambda storage, loc: storage)
model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
model_pos = model_backbone
model_pos.eval()
test_loader_params = {
    'batch_size': 1,
    'shuffle': False,
    'num_workers': 8,
    'pin_memory': True,
    'prefetch_factor': 4,
    'persistent_workers': True,
    'drop_last': False
}

vid = imageio.get_reader(opts.vid_path, 'ffmpeg')
fps_in = vid.get_meta_data()['fps']
vid_size = vid.get_meta_data()['size']
os.makedirs(opts.out_path, exist_ok=True)

if opts.pixel:
    # Keep relative scale with pixel coornidates
    wild_dataset = DetDataset(opts.json_path, clip_len=opts.clip_len, vid_size=vid_size, scale_range=None, focus=opts.focus)
else:
    # Scale to [-1,1]
    wild_dataset = DetDataset(opts.json_path, clip_len=opts.clip_len, scale_range=[1,1], focus=opts.focus)

test_loader = DataLoader(wild_dataset, **test_loader_params)

results_all = []
with torch.no_grad():
    for batch_input in tqdm(test_loader):
        N, T = batch_input.shape[:2]
        if torch.cuda.is_available():
            batch_input = batch_input.cuda()
        if args.no_conf: # Input 2D keypoints without confidence score.
            batch_input = batch_input[:, :, :, :2]
        if args.flip: # flip data for augmentation
            batch_input_flip = flip_data(batch_input)
            predicted_3d_pos_1 = model_pos(batch_input)
            predicted_3d_pos_flip = model_pos(batch_input_flip)
            predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip) # flip back
            predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2.0 # ensemble
        else:
            predicted_3d_pos = model_pos(batch_input)
        
        if args.rootrel: # 모든 프레임의 root 좌표를 0으로 설정
            predicted_3d_pos[:, :, 0, :] = 0
        else: # 첫 프레임에서 root의 z좌표만 0으로 설정
            predicted_3d_pos[:, 0, 0, 2] = 0
            pass
        
        if args.gt_2d:
            predicted_3d_pos[..., :2] = batch_input[..., :2]
        
        results_all.append(predicted_3d_pos.cpu().numpy())

results_all = np.hstack(results_all)
results_all = np.concatenate(results_all)

if opts.pixel:
    results_all = results_all * (min(vid_size) / 2.0)
    results_all[:, :, :2] = results_all[:, :, :2] + np.array(vid_size) / 2.0

filename = os.path.splitext(os.path.basename(opts.vid_path))

_, output_filename = os.path.split(opts.vid_path)
output_filename, _ = os.path.splitext(output_filename)
save_path = os.path.join(opts.out_path, output_filename+'.npy')
print('npy file saving...')
np.save(save_path, results_all)
