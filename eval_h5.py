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


import os, yaml, h5py, argparse
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import torch
import waffleiron
import numpy as np
from tqdm import tqdm
from waffleiron import Segmenter
from datasets import H5Dataset, Collate
from datasets.h5sf import CATEGORY_TO_INDEX, NAME_MAPPING_K2A, CAR, OTHER_VEHICLES, NAME_MAPPING_N2A

if __name__ == "__main__":
    # --- Arguments
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--ckpt", type=str, help="Path to checkpoint")
    parser.add_argument(
        "--path_dataset", type=str, help="Path to H5Dataset dataset"
    )
    parser.add_argument(
        "--num_votes", type=int, default=1, help="Number of test time augmentations"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--phase", required=True, help="val or test")
    parser.add_argument("--flow_mode", type=str, default='himu_seflowpp')
    args = parser.parse_args()
    assert args.num_votes % args.batch_size == 0

    # --- Load config file
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # --- SemanticKITTI (from https://github.com/PRBonn/semantic-kitti-api/blob/master/remap_semantic_labels.py)
    with open("./datasets/semantic-kitti.yaml") as stream:
        semkittiyaml = yaml.safe_load(stream)
    remapdict = semkittiyaml["learning_map_inv"]
    labeldict = semkittiyaml["labels"]
    # labeldict = semkittiyaml["nuslabels"]
    
    maxkey = max(remapdict.keys())
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(remapdict.keys())] = list(remapdict.values())

    # --- Dataloader
    tta = args.num_votes > 1
    print("Path: ", args.path_dataset)
    dataset = H5Dataset(
        rootdir=args.path_dataset,
        input_feat=config["embedding"]["input_feat"],
        voxel_size=config["embedding"]["voxel_size"],
        num_neighbors=config["embedding"]["neighbors"],
        dim_proj=config["waffleiron"]["dim_proj"],
        grids_shape=config["waffleiron"]["grids_size"],
        fov_xyz=config["waffleiron"]["fov_xyz"],
        phase=args.phase,
        tta=tta,
        flow_mode=args.flow_mode,
    )
    if args.num_votes > 1:
        new_list = []
        for f in dataset.im_idx:
            for v in range(args.num_votes):
                new_list.append(f)
        dataset.im_idx = new_list
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=Collate(),
    )
    args.num_votes = args.num_votes // args.batch_size

    # --- Build network
    net = Segmenter(
        input_channels=config["embedding"]["size_input"],
        feat_channels=config["waffleiron"]["nb_channels"],
        depth=config["waffleiron"]["depth"],
        grid_shape=config["waffleiron"]["grids_size"],
        nb_class=config["classif"]["nb_class"],
        drop_path_prob=config["waffleiron"]["drop"],
    )
    net = net.cuda()

    # --- Load weights
    ckpt = torch.load(args.ckpt, map_location="cuda:0")
    try:
        net.load_state_dict(ckpt["net"])
    except:
        # If model was trained using DataParallel or DistributedDataParallel
        state_dict = {}
        for key in ckpt["net"].keys():
            state_dict[key[len("module."):]] = ckpt["net"][key]
        net.load_state_dict(state_dict)
    net.compress()
    net.eval()

    # --- Re-activate droppath if voting
    if tta:
        for m in net.modules():
            if isinstance(m, waffleiron.backbone.DropPath):
                m.train()

    # --- Evaluation
    id_vote = 0
    tmp_seg_name = f'seg_{args.flow_mode}'
    # CAR+OTHER_VEHICLES extract their index in CATEGORY_TO_INDEX
    valid_index_ = [CATEGORY_TO_INDEX[l] for l in CAR + OTHER_VEHICLES]
    for it, batch in enumerate(tqdm(loader, bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:50}{r_bar}")):
        if it>10:
            break
        # Reset vote
        if id_vote == 0:
            vote = None

        # Network inputs
        feat = batch["feat"].cuda(non_blocking=True)
        labels = batch["labels_orig"].cuda(non_blocking=True)
        batch["upsample"] = [up.cuda(non_blocking=True) for up in batch["upsample"]]
        cell_ind = batch["cell_ind"].cuda(non_blocking=True)
        occupied_cell = batch["occupied_cells"].cuda(non_blocking=True)
        neighbors_emb = batch["neighbors_emb"].cuda(non_blocking=True)
        net_inputs = (feat, cell_ind, occupied_cell, neighbors_emb)

        # Get prediction
        with torch.autocast("cuda", enabled=True):
            with torch.inference_mode():
                # Get prediction
                out = net(*net_inputs)
                for b in range(out.shape[0]):
                    temp = out[b, :, batch["upsample"][b]].T
                    if vote is None:
                        vote = torch.softmax(temp, dim=1)
                    else:
                        vote += torch.softmax(temp, dim=1)
        id_vote += 1

        # Save prediction
        if id_vote == args.num_votes:
            id_vote = 0
            # Convert label
            pred_label = (
                vote.max(1)[1] + 1
            )  # Shift by 1 because of ignore_label at index 0

            ## KITTI:
            label_ = pred_label.cpu().numpy().reshape(-1).astype(np.uint32)
            upper_half = label_ >> 16  # get upper half for instances
            lower_half = label_ & 0xFFFF  # get lower half for semantics
            label = remap_lut[lower_half]  # do the remapping of semantics
            res_sem_ = [CATEGORY_TO_INDEX[NAME_MAPPING_K2A[labeldict[l]]] for l in label]
            
            ## nuScenes:
            # label = pred_label.cpu().numpy().reshape(-1).astype(np.uint8)
            # res_sem_ = [CATEGORY_TO_INDEX[NAME_MAPPING_N2A[labeldict[l]]] for l in label]

            # Save result
            assert batch["filename"][0] == batch["filename"][-1]
            scene_id, timestamp = batch["filename"][0].split(":")
            # # print("Scene ID: ", scene_id, "Timestamp: ", timestamp)
            with h5py.File(os.path.join(args.path_dataset, f'{scene_id}.h5'), 'r+') as f:    
                key = str(timestamp)
                if 'flow_category_indices' not in f[key]:
                    continue
                valid_class = np.isin(f[key]['flow_category_indices'][:], valid_index_)
                gm = f[key]['ground_mask'][:]

                # gm removed in seg network
                # res_sem = np.zeros_like(valid_class)
                # res_sem[~gm] = np.array(res_sem_)

                # no gm remove in seg network
                seg_valid = valid_class & ~gm
                res_sem = np.array(res_sem_)
                # res_sem[~seg_valid] = 0

                if 'seg_valid' not in f[key]:
                    f[key].create_dataset('seg_valid', data=seg_valid)
                
                if tmp_seg_name in f[key]:
                    # del f[key][tmp_seg_name]
                    continue
                # else:
                #     f[key].create_dataset(tmp_seg_name, data=res_sem)
                f[key].create_dataset(tmp_seg_name, data=res_sem)

    print(f"Segmentation result name: {tmp_seg_name}, Please check the h5 file for the result.")