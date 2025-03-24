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

import pickle, h5py, os, sys
import numpy as np
from .pc_dataset import PCDataset

from av2.datasets.sensor.constants import AnnotationCategories
from typing import Final
SCENE_FLOW_DYNAMIC_THRESHOLD: Final = 0.05
SWEEP_PAIR_TIME_DELTA: Final = 0.1
CLOSE_DISTANCE_THRESHOLD: Final = 35.0

CATEGORY_TO_INDEX: Final = {
    **{"NONE": 0},
    **{k.value: i + 1 for i, k in enumerate(AnnotationCategories)},
}
INDEX_TO_CATEGORY: Final = {v: k for k, v in CATEGORY_TO_INDEX.items()}
NAME_MAPPING_K2A = {
    'outlier': 'NONE',
    'unlabeled': 'NONE',
    'car': 'REGULAR_VEHICLE',
    'bicycle': 'BICYCLE',
    'motorcycle': 'MOTORCYCLE',
    'truck': 'TRUCK',
    'other-vehicle': 'LARGE_VEHICLE',
    'person': 'PEDESTRIAN',
    'bicyclist': 'BICYCLIST',
    'motorcyclist': 'MOTORCYCLIST',
    'road': 'NONE',
    'parking': 'NONE',
    'sidewalk': 'NONE',
    'other-ground': 'NONE',
    'building': 'NONE',
    'fence': 'NONE',
    'vegetation': 'NONE',
    'trunk': 'NONE',
    'terrain': 'NONE',
    'pole': 'NONE',
    'traffic-sign': 'SIGN',
}    
PEDESTRIAN_CATEGORIES = ["PEDESTRIAN", "STROLLER", "WHEELCHAIR", "OFFICIAL_SIGNALER"]
WHEELED_VRU = [
    "BICYCLE",
    "BICYCLIST",
    "MOTORCYCLE",
    "MOTORCYCLIST",
    "WHEELED_DEVICE",
    "WHEELED_RIDER",
]
CAR = ["REGULAR_VEHICLE"]
OTHER_VEHICLES = [
    "BOX_TRUCK",
    "LARGE_VEHICLE",
    "RAILED_VEHICLE",
    "TRUCK",
    "TRUCK_CAB",
    "VEHICULAR_TRAILER",
    "ARTICULATED_BUS",
    "BUS",
    "SCHOOL_BUS",
]
class H5Dataset(PCDataset):
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
