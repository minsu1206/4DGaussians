#!/bin/bash
# scene=dnerf/jumpingjacks
# python train.py -s ../dataset/$scene --port 6017 --expname "$scene" --configs arguments/$scene.py 
# python render.py --model_path "output/$scene"  --skip_train --configs arguments/$scene.py
# python metrics.py --model_path "output/$scene"

dataset=ST_NeRF
scene=walking

# CUDA_VISIBLE_DEVICES=2 python scripts/downsample_point.py /workspace/lustre/datasets/nerf_team/${scene}/fused.ply /workspace/lustre/datasets/nerf_team/${scene}/points3D_downsample2.ply
CUDA_VISIBLE_DEVICES=1 python train.py -s /workspace/lustre/datasets/nerf_team/$scene --port 6017 --expname "$scene" --configs arguments/$dataset/$scene.py --sparse
CUDA_VISIBLE_DEVICES=1 python render.py --model_path "output/$scene"  --skip_train --configs arguments/$dataset/$scene.py
CUDA_VISIBLE_DEVICES=1 python metrics.py --model_path "output/$scene"

scene=taekwondo
# CUDA_VISIBLE_DEVICES=2 python scripts/downsample_point.py /workspace/lustre/datasets/nerf_team/${scene}/fused.ply /workspace/lustre/datasets/nerf_team/${scene}/points3D_downsample2.ply
CUDA_VISIBLE_DEVICES=1 python train.py -s /workspace/lustre/datasets/nerf_team/$scene --port 6019 --expname "$scene" --configs arguments/$dataset/$scene.py --sparse
CUDA_VISIBLE_DEVICES=1 python render.py --model_path "output/$scene"  --skip_train --configs arguments/$dataset/$scene.py
CUDA_VISIBLE_DEVICES=1 python metrics.py --model_path "output/$scene"