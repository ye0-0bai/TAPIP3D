# Copyright (c) TAPIP3D team(https://tapip3d.github.io/)

from concurrent.futures import ThreadPoolExecutor
import shlex
import tap
import torch
from typing import Optional, Tuple
from pathlib import Path
from datetime import datetime
from einops import repeat ,rearrange
from utils.common_utils import setup_logger
import logging
from annotation.megasam import MegaSAMAnnotator
import numpy as np
import cv2
from datasets.data_ops import _filter_one_depth

from utils.inference_utils import load_model, read_video, inference, get_grid_queries, resize_depth_bilinear

import os
import pickle
from tqdm import tqdm
import torch_fpsample

logger = logging.getLogger(__name__)

DEFAULT_QUERY_GRID_SIZE = 32

class Arguments(tap.Tap):
    input_path: str = "/mnt/nas_24/wangwq/datasets/calvin/refactored/task_ABC_D/training/static_camera_observation/0410576_0410640.npz"
    device: str = "cuda"
    num_iters: int = 6
    support_grid_size: int = 16
    num_threads: int = 8
    resolution_factor: int = 2
    vis_threshold: Optional[float] = 0.9
    checkpoint: Optional[str] = "/data/repository/TAPIP3D/checkpoints/tapip3d_final.pth"
    output_dir: str = "/mnt/nas_24/wangwq/datasets/calvin/refactored/task_ABC_D/training/tapip3d_output"
    depth_model: str = "moge"


def convert_depth_to_point_cloud(depth:np.ndarray, intrinsic:np.ndarray, scale:float=1.0) -> np.ndarray:
    """
    Convert depth image(s) to 3D point cloud(s) using camera intrinsics.

    Args:
        depth (np.ndarray)
        intrinsic (np.ndarray)
        scale (float, optional): scaling factor to convert depth values to meters. Defaults to 1.0.

    Returns:
        np.ndarray: point cloud(s)
    """

    if depth.ndim == 2:
        batched_input = False
        depth = depth[None,...]
        intrinsic = intrinsic[None,...]
    elif depth.ndim == 3:
        batched_input = True
        if intrinsic.ndim == 2:
            intrinsic = intrinsic[None,...]
    else:
        raise NotImplementedError
        
    H, W = depth.shape[1], depth.shape[2]
    u,v = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    
    u = u.astype(np.float32)
    v = v.astype(np.float32)
    depth = depth.astype(np.float32)
    intrinsic = intrinsic.astype(np.float32)
    
    Z = depth / scale
    X = (u[None,...] - intrinsic[:,0,2][:,None,None]) * Z / intrinsic[:,0,0][:,None,None]
    Y = (v[None,...] - intrinsic[:,1,2][:,None,None]) * Z / intrinsic[:,1,1][:,None,None]
    
    point_cloud = np.stack((X, Y, Z), axis=-1)
    
    if batched_input:
        return point_cloud
    else:
        return point_cloud[0]


def prepare_inputs(input_path: str, inference_res: Tuple[int, int], support_grid_size: int, num_threads: int = 8, device: str = "cpu"):
    
    data = np.load(input_path)
    
    video = data['rgb']
    depths = data['depth']
    K = [
        [1143.0051803588867, 0.0,                100.0],
        [0.0,                1143.0051803588867, 100.0],
        [0.0,                0.0,                1.0  ]
    ]
    intrinsics = repeat(np.array(K), 'h w -> t h w', t=len(video))
    
    extrinsics = repeat(np.identity(4), 'h w -> t h w', t=len(video))
    query_point = convert_depth_to_point_cloud(depths, intrinsics)
    query_point = torch.from_numpy(query_point).float()
    query_point = rearrange(query_point, 't h w c -> t (h w) c')
    query_point = torch_fpsample.sample(query_point, k=1024, h=7)[0]
    query_point = torch.concatenate([torch.zeros_like(query_point[...,:1], dtype=query_point.dtype, device=query_point.device), query_point], axis=-1)
    query_point = query_point.to(device)

    _original_res = video.shape[1:3]
    
    intrinsics[:, 0, :] *= (inference_res[1] - 1) / (_original_res[1] - 1)
    intrinsics[:, 1, :] *= (inference_res[0] - 1) / (_original_res[0] - 1)

    # resize & remove edges
    with ThreadPoolExecutor(num_threads) as executor:
        video_futures = [executor.submit(cv2.resize, rgb, (inference_res[1], inference_res[0]), interpolation=cv2.INTER_LINEAR) for rgb in video]
        depths_futures = [executor.submit(resize_depth_bilinear, depth, (inference_res[1], inference_res[0])) for depth in depths]
        
        video = np.stack([future.result() for future in video_futures])
        depths = np.stack([future.result() for future in depths_futures])

        depths_futures = [executor.submit(_filter_one_depth, depth, 0.08, 15, intrinsic) for depth, intrinsic in zip(depths, intrinsics)]
        depths = np.stack([future.result() for future in depths_futures])

    video = (torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0).to(device)
    depths = torch.from_numpy(depths).float().to(device)
    intrinsics = torch.from_numpy(intrinsics).float().to(device)
    extrinsics = torch.from_numpy(extrinsics).float().to(device)
    # query_point = torch.from_numpy(query_point).float().to(device)

    return video, depths, intrinsics, extrinsics, query_point, support_grid_size


if __name__ == "__main__":
    setup_logger()
    args = Arguments().parse_args()
    
    logger.info(f'Processing {args.input_path}...')
    
    # output_dir = Path(args.output_dir) / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    # output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(args.checkpoint)
    model.to(args.device)

    inference_res = (int(model.image_size[0] * np.sqrt(args.resolution_factor)), int(model.image_size[1] * np.sqrt(args.resolution_factor)))
    model.set_image_size(inference_res)

    # Prepare inputs
    video, depths, intrinsics, extrinsics, query_point, support_grid_size = prepare_inputs(
        input_path=args.input_path, 
        inference_res=inference_res, 
        support_grid_size=args.support_grid_size,
        num_threads=args.num_threads,
        device=args.device
    )

    # Run inference
    length = len(video)
    all_coords = {}
    all_visibs = {}
    for start_idx in tqdm(range(0, length-1, 8), ncols=100):
        
        end_idx = min(length, start_idx + 32)
        
        with torch.autocast(args.device, dtype=torch.bfloat16):
            coords, visibs = inference(
                model=model,
                video=video[start_idx:end_idx],
                depths=depths[start_idx:end_idx],
                intrinsics=intrinsics[start_idx:end_idx],
                extrinsics=extrinsics[start_idx:end_idx],
                query_point=query_point[start_idx],
                num_iters=args.num_iters,
                grid_size=support_grid_size,
            )
        all_coords[start_idx] = coords.cpu().numpy().copy()
        all_visibs[start_idx] = visibs.cpu().numpy().copy()
        
    data = {
        'coords' : all_coords,
        'visibs' : all_visibs
    }
    
    file_name = args.input_path.split('/')[-1]
    
    with open(os.path.join(args.output_dir, file_name), 'wb') as f:
        pickle.dump(data, f)
        
    logger.info(f'Done.')
    
    
