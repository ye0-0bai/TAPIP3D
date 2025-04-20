import argparse
import glob
import os
# import matplotlib.pyplot as plt
from timeit import default_timer as timer
import cv2
import torch
from video_depth_anything.video_depth import VideoDepthAnything
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--img-path', type=str)
  parser.add_argument('--outdir', type=str, default='./vis_depth')
  parser.add_argument('--load-from', type=str, required=True)
  parser.add_argument('--encoder', type=str, default='vitl')

  args = parser.parse_args()

  margin_width = 50
  caption_height = 60

  font = cv2.FONT_HERSHEY_SIMPLEX
  font_scale = 1
  font_thickness = 2
  model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
  assert args.encoder in ['vits', 'vitb', 'vitl']
  depth_anything = VideoDepthAnything(**model_configs[args.encoder])
  total_params = sum(param.numel() for param in depth_anything.parameters())
  print('Total parameters: {:.2f}M'.format(total_params / 1e6))

  depth_anything.load_state_dict(
      torch.load(args.load_from, map_location='cpu'), strict=True
  )

  depth_anything.eval()
  depth_anything.to(DEVICE)

  if os.path.isfile(args.img_path):
    if args.img_path.endswith('txt'):
      with open(args.img_path, 'r') as f:
        filenames = f.read().splitlines()
    else:
      filenames = sorted(glob.glob(os.path.join(args.img_path, '*.png')))

  filenames = sorted(glob.glob(os.path.join(args.img_path, '*.png')))
  filenames += sorted(glob.glob(os.path.join(args.img_path, '*.jpg')))

  final_results = []
  raw_images = []
  for filename in tqdm(filenames):
    raw_image = cv2.imread(filename)[..., :3]
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    raw_images.append(raw_image)
    h, w = raw_image.shape[:2]
  raw_images = np.array(raw_images)
  depth_list, _ = depth_anything.infer_video_depth(raw_images, target_fps=None, device=DEVICE)
  
  for filename, depth in zip(filenames, depth_list):
    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    depth_npy = np.float32(depth)

    os.makedirs(os.path.join(args.outdir), exist_ok=True)
    np.save(
        os.path.join(args.outdir, filename.split('/')[-1][:-4] + '.npy'),
        depth_npy,
    )
