#!/bin/bash
#SBATCH -J segkitti
#SBATCH --mem 360GB
#SBATCH --gres gpu:8
#SBATCH --cpus-per-task 46
#SBATCH --constrain "galadriel"
#SBATCH --output /Midgard/home/qingwen/logs/seg/%J.out
#SBATCH --error  /Midgard/home/qingwen/logs/seg/%J.err

# galadriel|eowyn|balrog|khazadum
PYTHON=/Midgard/home/qingwen/miniforge3/envs/waffleiron/bin/python
cd /local_storage/users/qingwen/waffleiron
git pull
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Midgard/home/qingwen/miniforge3/lib

$PYTHON launch_train.py \
--dataset semantic_kitti \
--path_dataset /local_storage/datasets/qingwen/data/kitti/data_odometry_velodyne \
--log_path ./logs/WaffleIron-48-256__kitti \
--config ./configs/WaffleIron-48-256_kitti_noin.yaml \
--multiprocessing-distributed \
--fp16
