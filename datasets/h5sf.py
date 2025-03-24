# Copyright 2022 - Valeo Comfort and Driving Assistance - Gilles Puy @ valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import yaml, pickle
import torch
import warnings, h5py, os, sys
import numpy as np
from glob import glob
from tqdm import tqdm
import utils.transforms as tr
from .pc_dataset import PCDataset

class H5Dataset(PCDataset):
    CLASS_NAME = [
        "car",  # 0
        "bicycle",  # 1
        "motorcycle",  # 2
        "truck",  # 3
        "other-vehicle",  # 4
        "person",  # 5
        "bicyclist",  # 6
        "motorcyclist",  # 7
        "road",  # 8
        "parking",  # 9
        "sidewalk",  # 10
        "other-ground",  # 11
        "building",  # 12
        "fence",  # 13
        "vegetation",  # 14
        "trunk",  # 15
        "terrain",  # 16
        "pole",  # 17
        "traffic-sign",  # 18
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.phase in ['test', 'val'], "H5Dataset only supports validation phase now."
        # eval_index_file = os.path.join(self.rootdir, 'index_eval.pkl')
        eval_index_file = os.path.join(self.rootdir, 'index_total.pkl')
        if not os.path.exists(eval_index_file):
            raise Exception(f"No eval index file found! Please check {self.rootdir}")
        with open(eval_index_file, 'rb') as f:
            self.im_idx = pickle.load(f)

    def __len__(self):
        return len(self.im_idx)

    def __load_pc_internal__(self, index):
        # Load point cloud
        # pc = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        scene_id, timestamp = self.im_idx[index]
        with h5py.File(os.path.join(self.rootdir, f'{scene_id}.h5'), 'r') as f:
            key = str(timestamp)
            pc = f[key]['lidar'][:]
        # set intensity to 0 as we don't know if the scale is the same.
        pc[:, 3] = 0
        # print(pc.shape) # testing
        # Extract Label
        labels = np.zeros((pc.shape[0], 1), dtype=np.int32)
        labels_inst = np.zeros((pc.shape[0], 1), dtype=np.int32)

        # Map ignore index 0 to 255
        labels = labels[:, 0] - 1
        labels[labels == -1] = 255

        return pc, labels, labels_inst[:, 0]

    def load_pc(self, index):
        pc, labels, labels_inst = self.__load_pc_internal__(index)
        scene_id, timestamp = self.im_idx[index]
        return pc, labels, scene_id+":"+timestamp
