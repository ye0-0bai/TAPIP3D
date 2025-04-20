import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
from pathlib import Path

from PIL import Image
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).absolute().parents[1]))

from typing import *
import itertools
import json
import warnings

import trimesh
import trimesh.visual
# import click

from moge.model import MoGeModel
from moge.utils.io import save_glb, save_ply
from moge.utils.vis import colorize_depth, colorize_normal
import utils3d

file_path = Path(__file__).parent

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MoGe')
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    parser.add_argument('--fov_x', type=str, default='')
    
    args = parser.parse_args()
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Load model and preprocessing transform
    model = MoGeModel.from_pretrained('Ruicheng/moge-vitl').to(DEVICE).eval()
    
    filenames = sorted(glob.glob(os.path.join(args.img_path, '*.png')))
    filenames += sorted(glob.glob(os.path.join(args.img_path, '*.jpg')))

    os.makedirs(args.outdir, exist_ok=True)
    
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        image = cv2.cvtColor(cv2.imread(str(filename)), cv2.COLOR_BGR2RGB)
        image_tensor = torch.tensor(image / 255, dtype=torch.float32, device=DEVICE).permute(2, 0, 1)

        if args.fov_x:
            output = model.infer(image_tensor, fov_x=torch.tensor(float(args.fov_x), device=DEVICE), force_projection=False)
        else:
            output = model.infer(image_tensor, force_projection=False)

        raw_mask = output['raw_mask'].clone()
        raw_mask[~torch.isfinite(output['raw_depth'])] = False

        raw_inv_depth = 1. / torch.clamp(output['raw_depth'], 1e-4, 1e4)
        raw_inv_depth[~raw_mask] = 0.

        raw_depth = output['raw_depth'].clone()
        raw_depth[~raw_mask] = 0.
        output['raw_depth'] = raw_depth
        
        # to_save = {
        #     "raw_inverse_depth": raw_inv_depth.cpu().numpy(),
        #     "raw_points": output["raw_points"].cpu().numpy(),
        #     "fov_x": output["fov_x"].item(),
        #     "raw_depth": output["raw_depth"].cpu().numpy(),
        #     "raw_mask": output["raw_mask"].cpu().numpy(),
        # }
            
        # np.savez(os.path.join(args.outdir, filename.split('/')[-1][:-4] + '.npz'), **to_save)
        np.save(os.path.join(args.outdir, filename.split('/')[-1][:-4] + '.npy'), raw_inv_depth.squeeze(0).cpu().numpy().astype(np.float32))