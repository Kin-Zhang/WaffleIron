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


NAME_MAPPING_N2A = {
    'ignore': 'NONE',
    'barrier': 'NONE',
    'bicycle': 'BICYCLE',
    'bus': 'BUS',
    'car': 'REGULAR_VEHICLE',
    'construction_vehicle': 'LARGE_VEHICLE',
    'motorcycle': 'MOTORCYCLE',
    'pedestrian': 'PEDESTRIAN',
    'traffic_cone': 'NONE',
    'trailer': 'VEHICULAR_TRAILER',
    'truck': 'TRUCK',
    'driveable_surface': 'NONE',
    'other_flat': 'NONE',
    'sidewalk': 'NONE',
    'terrain': 'NONE',
    'manmade': 'NONE',
    'vegetation': 'NONE',
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
        eval_index_file = os.path.join(self.rootdir, 'index_eval.pkl') if self.phase == 'val' else os.path.join(self.rootdir, 'index_total.pkl')
        # eval_index_file = os.path.join(self.rootdir, 'index_total.pkl')
        if not os.path.exists(eval_index_file):
            raise Exception(f"No eval index file found! Please check {self.rootdir}")
        with open(eval_index_file, 'rb') as f:
            self.im_idx = pickle.load(f)

        self.scene_id_bounds = {}  # 存储每个scene_id的最大最小timestamp和位置
        for idx, (scene_id, timestamp) in enumerate(self.im_idx):
            if scene_id not in self.scene_id_bounds:
                self.scene_id_bounds[scene_id] = {
                    "min_timestamp": timestamp,
                    "max_timestamp": timestamp,
                    "min_index": idx,
                    "max_index": idx
                }
            else:
                bounds = self.scene_id_bounds[scene_id]
                # 更新最小timestamp和位置
                if timestamp < bounds["min_timestamp"]:
                    bounds["min_timestamp"] = timestamp
                    bounds["min_index"] = idx
                # 更新最大timestamp和位置
                if timestamp > bounds["max_timestamp"]:
                    bounds["max_timestamp"] = timestamp
                    bounds["max_index"] = idx

    def __len__(self):
        return len(self.im_idx)
    
    def egopts_mask(self, pts, min_bound=[-9.5, -3/2, 0], max_bound=[5, 2.760004/2, 5]):
        """
        Input:
            pts: (N, 3)
        Output:
            mask: (N, ) indicate the points that are outside the egopts
        """
        mask = ((pts[:, 0] > min_bound[0]) & (pts[:, 0] < max_bound[0])
                & (pts[:, 1] > min_bound[1]) & (pts[:, 1] < max_bound[1])
                & (pts[:, 2] > min_bound[2]) & (pts[:, 2] < max_bound[2]))
        return ~mask
            
    def __load_pc_internal__(self, index):
        # Load point cloud
        # pc = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        scene_id, timestamp = self.im_idx[index]
        with h5py.File(os.path.join(self.rootdir, f'{scene_id}.h5'), 'r') as f:
            key = str(timestamp)
            pc = f[key]['lidar'][:]
            ego_mask = ~self.egopts_mask(pc, min_bound=[-1.5, -1.5, -2.0], max_bound=[1.5, 1.5, 2.0])
            pc[:, 2] -= 1.73 # add height offset
            # remove ground point?
            # gm = f[key]['ground_mask'][:]
            # pc = pc[~gm]
            if self.flow_mode in f[key]:
                # print('Using flow mode:', self.flow_mode)
                pose0 = f[key]['pose'][:]
                scene_id1, timestamp1 = self.im_idx[index+1]
                pose1 = f[str(timestamp1)]['pose'][:]
                ego_pose = np.linalg.inv(pose1) @ pose0
                pose_flow = pc[:, :3] @ ego_pose[:3, :3].T + ego_pose[:3, 3] - pc[:, :3]
                dt0_raw = f[key]['lidar_dt'][:]
                flow = f[key][self.flow_mode][:] - pose_flow
                flow[ego_mask] = 0
                dt0 = max(dt0_raw) - dt0_raw
                ref_pc = pc[:,:3] + (flow/0.1) * dt0[:, None]

                # remove ground point?
                # flow = f[key][self.flow_mode][:][~gm] - pose_flow
                # dt0 = max(dt0_raw) - dt0_raw # we want to see the last frame. dts: (N,1) Nanosecond offsets _from_ the start of the sweep.
                # ref_pc = pc[:,:3] + (flow/0.1) * dt0[:, None][~gm]
            else:
                ref_pc = pc[:,:3]
        pc[:, :3] = ref_pc
        pc[:, 3]  = 0       # set intensity to 0 as we don't know if the scale is the same.

        # Extract Label
        labels = np.zeros((pc.shape[0], 1), dtype=np.int32)
        labels_inst = np.zeros((pc.shape[0], 1), dtype=np.int32)

        # Map ignore index 0 to 255
        labels = labels[:, 0] - 1
        labels[labels == -1] = 255

        return pc, labels, labels_inst[:, 0]

    def load_pc(self, index):
        pc, labels, labels_inst = self.__load_pc_internal__(index)
        # # double check whether the point cloud is undistorted one.
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pc[:,:3])
        # o3d.visualization.draw_geometries([pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)])
        scene_id, timestamp = self.im_idx[index]
        return pc, labels, scene_id+":"+timestamp
