"""
# Created: 2023-12-31 22:19
# LastEdit: 2024-01-12 18:46
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Kin ZHANG  (https://kin-zhang.github.io/)

# Description:
#   Delete useless data in h5 files, and repack them.

# Example Running:
python tests/0_delete_useless_data.py --data_dir /home/kin/data/av2/preprocess/sensor/vis --fixed_dir /home/kin/data/av2/preprocess/sensor/vis_fixed --del_lists "[flow_est]"
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import multiprocessing
from pathlib import Path
from multiprocessing import Pool, current_process
from typing import Optional, Tuple, Dict, Union, Final
from tqdm import tqdm
import fire, time, h5py


def process_log(data_dir: Path, log: str, del_lists = ['flow', 'flow_is_valid', 'flow_category_indices'], n: Optional[int] = None) :
    with h5py.File(os.path.join(data_dir, f'{log}'), 'r+') as f:
        for k in list(f.keys()):
            if len(f[k]) == 0:
                del f[k]
                # print(f"Delete {k}")
            for del_name in del_lists:
                if del_name in f[k]:
                    del f[k][del_name]
                    # print(f"Delete {del_name} in {k}")

def proc(x, ignore_current_process=False):
    if not ignore_current_process:
        current=current_process()
        pos = current._identity[0]
    else:
        pos = 1
    process_log(*x, n=pos)
    
def delete(
    data_dir: str ="/home/kin/data/av2/seflow_preprocess/lidar/train",
    fixed_dir: str ="/home/kin/data/av2/seflow_preprocess/lidar/train_fixed", # no need for delete
    del_lists: list = ['flow', 'flow_is_valid', 'flow_category_indices'],
    nproc: int = (multiprocessing.cpu_count() - 1)
):
    # python tests/0_delete_useless_data.py --data_dir /proj/berzelius-2023-154/users/x_qinzh/av2/seflow_preprocess/lidar/train
    # NOTE(Qingwen): if you don't want to all data_dir, then change here: logs = logs[:10] only 10 scene.
    logs = os.listdir(data_dir)
    args = sorted([(data_dir, log, del_lists) for log in logs if log.endswith('.h5')])
    print(f'Using {nproc} processes for deleting useless data')
    # for debug
    # for x in tqdm(args):
    #     proc(x, ignore_current_process=True)
    #     break
    if nproc <= 1:
        for x in tqdm(args):
            proc(x, ignore_current_process=True)
    else:
        with Pool(processes=nproc) as p:
            res = list(tqdm(p.imap_unordered(proc, args), total=len(logs), ncols=100))

def repack_single(x):
    os.system(f"h5repack {os.path.join(x[0], x[-1])} {os.path.join(x[1], x[-1])}")

def repack(
    data_dir: str ="/home/kin/data/Scania/preprocess/val_v1",
    fixed_dir: str ="/home/kin/data/Scania/preprocess/val_v1_repack",
    del_lists = None,
    nproc: int = (multiprocessing.cpu_count() - 1)
):
    logs = os.listdir(data_dir)
    os.makedirs(fixed_dir, exist_ok=True)
    args = sorted([(data_dir, fixed_dir, log) for log in logs if log.endswith('.h5')])
    print(f'Using {nproc} processes for repacking the h5 files')
    if nproc <= 1:
        for x in tqdm(args):
            repack_single(x, ignore_current_process=True)
    else:
        with Pool(processes=nproc) as p:
            res = list(tqdm(p.imap_unordered(repack_single, args), total=len(logs), ncols=100))

if __name__ == '__main__':
    start_time = time.time()
    # fire.Fire(delete)
    fire.Fire(repack)
    print(f"\nTime used: {(time.time() - start_time)/60:.2f} mins")