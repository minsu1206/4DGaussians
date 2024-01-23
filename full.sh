#!/bin/bash
# scene=dnerf/jumpingjacks
# python train.py -s ../dataset/$scene --port 6017 --expname "$scene" --configs arguments/$scene.py 
# python render.py --model_path "output/$scene"  --skip_train --configs arguments/$scene.py
# python metrics.py --model_path "output/$scene"

dataset=ST_NeRF
scene=taekwondo

# python scripts/downsample_point.py data/dynerf/sear_steak/colmap/dense/workspace/fused.ply data/dynerf/sear_steak/points3D_downsample2.ply
# python train.py -s ../dataset/$dataset/$scene --port 6017 --expname "$scene" --configs arguments/$dataset/$scene.py 
python render.py --model_path "output/$scene"  --skip_train --configs arguments/$dataset/$scene.py
python metrics.py --model_path "output/$scene"