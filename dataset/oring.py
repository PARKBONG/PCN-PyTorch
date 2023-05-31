import sys
sys.path.append('.')

import os
import random

import torch
import torch.utils.data as data
import numpy as np
import open3d as o3d

TEST_START = 0
TEST_END = 99
VALID_START = 100
VALID_END = 299
TRAIN_START = 300
TRAIN_END = 1998

# oring_file_dict = {"005_03" : {"filemax_index": 5, 1: 2000, 2: 2000, 3: 2000, 4: 2000, 5: 2000},
#                    "005_05" : {"filemax_index": 5, 1: 2000, 2: 2000, 3: 2000, 4: 2000, 5: 2000},
#                    "005_07" : {"filemax_index": 5, 1: 2000, 2: 2000, 3: 2000, 4: 2000, 5: 2000},
#                    "01_03"  : {"filemax_index": 5, 1: 2000, 2: 2000, 3: 2000, 4: 2000, 5: 2000},
#                    "01_05"  : {"filemax_index": 5, 1: 2000, 2: 2000, 3: 2000, 4: 2000, 5: 2000},
#                    "01_07"  : {"filemax_index": 5, 1: 2000, 2: 2000, 3: 2000, 4: 2000, 5: 2000},
#                    "015_03" : {"filemax_index": 7, 1:2000, 2:2000, 3:2000, 4:1000, 5: 1000, 6: 1000, 7: 1000},
#                    "015_05" : {"filemax_index": 7, 1:2000, 2:2000, 3:2000, 4:1000, 5: 1000, 6: 1000, 7: 1000},
#                    "015_07" : {"filemax_index": 8, 1:2000, 2:2000, 3:1000, 4:1000, 5: 1000, 6: 1000, 7: 1000, 8: 1000},
#                    }

oring_file_dict = {"005_03" : {"filemax_index": 5, 1: 2000, 2: 2000, 3: 2000, 4: 2000, 5: 2000},
                   "005_05" : {"filemax_index": 5, 1: 2000, 2: 2000, 3: 2000, 4: 2000, 5: 2000},
                   "005_07" : {"filemax_index": 1, 1: 2000},
                   "01_03"  : {"filemax_index": 5, 1: 2000, 2: 2000, 3: 2000, 4: 2000, 5: 2000},
                   "01_05"  : {"filemax_index": 5, 1: 2000, 2: 2000, 3: 2000, 4: 2000, 5: 2000},
                #    "01_07"  : {"filemax_index": 5, 1: 2000, 2: 2000, 3: 2000, 4: 2000, 5: 2000},
                   "015_03" : {"filemax_index": 3, 1:2000, 2:2000, 3:2000},
                   "015_05" : {"filemax_index": 7, 1:2000, 2:2000, 3:2000, 4:1000, 5: 1000, 6: 1000, 7: 1000},
                   "015_07" : {"filemax_index": 8, 1:2000, 2:2000, 3:1000, 4:1000, 5: 1000, 6: 1000, 7: 1000, 8: 1000},
                   }

# oring_list = ["005_03", "005_05", "005_07", "01_03", "01_05", "01_07", "015_03", "015_05", "015_07"]
# oring_list = ["005_03", "005_05", "005_07", "01_03", "01_05", "015_03", "015_05", "015_07"]
oring_list = list(oring_file_dict.keys())


class Oring(data.Dataset):
    """
    ShapeNet dataset in "PCN: Point Completion Network". It contains 28974 training
    samples while each complete samples corresponds to 8 viewpoint partial scans, 800
    validation samples and 1200 testing samples.
    """
    
    def __init__(self, dataroot="data/ORING", split="train", category=None):
        assert split in ['train', 'valid', 'test', 'test_novel'], "split error value!"
    
        self.dataroot = dataroot
        self.split = split
        self.category = category

        self.partial_paths, self.complete_paths = self._load_data()
    
    def __getitem__(self, index):
        partial_path = self.partial_paths[index]
        complete_path = self.complete_paths[index]

        partial_pc = self.random_sample(self.read_partial_point_cloud(partial_path), 512)
        complete_pc = self.random_sample(self.read_complete_point_cloud(complete_path), 1024)

        return torch.from_numpy(partial_pc), torch.from_numpy(complete_pc)

    def __len__(self):
        return len(self.complete_paths)

    def _load_data(self):
        partial_paths, complete_paths = list(), list()

        if self.split == 'train':
            for oring in oring_list:
                for fileindex in range(1, oring_file_dict[oring]["filemax_index"]+1):
                    path = f"{self.dataroot}/{oring}/{oring}_{fileindex}"
                    for model_id in range(1, oring_file_dict[oring][fileindex]+1):
                        if (model_id % 4):
                            partial_paths.append(os.path.join(path, 'segpcd_data', '{}.npy'.format(model_id)))
                            complete_paths.append(os.path.join(path, 'gt_data_xyzn', '{}.xyzn'.format(model_id)))
        elif self.split == 'valid':
            for oring in oring_list:
                for fileindex in range(1, oring_file_dict[oring]["filemax_index"]+1):
                    path = f"{self.dataroot}/{oring}/{oring}_{fileindex}"
                    for model_id in range(1, oring_file_dict[oring][fileindex]+1):
                        if not (model_id % 4):
                            partial_paths.append(os.path.join(path, 'segpcd_data', '{}.npy'.format(model_id)))
                            complete_paths.append(os.path.join(path, 'gt_data_xyzn', '{}.xyzn'.format(model_id)))
        elif self.split == 'test':
            for oring in oring_list:
                for fileindex in range(1, oring_file_dict[oring]["filemax_index"]+1):
                    path = f"{self.dataroot}/{oring}/{oring}_{fileindex}"
                    for model_id in range(TEST_START, TEST_END+1):
                        partial_paths.append(os.path.join(path, 'segpcd_data', '{}.npy'.format(model_id)))
                        complete_paths.append(os.path.join(path, 'gt_data_xyzn', '{}.xyzn'.format(model_id)))
        else:
            raise NotImplementedError

        return partial_paths, complete_paths
    
    def read_partial_point_cloud(self, path):
        pc = np.load(path,allow_pickle=True)
        return pc.astype(np.float32)   
    
    def read_complete_point_cloud(self, path):
        pc = o3d.io.read_point_cloud(path)
        return np.array(pc.points, np.float32)
    
    def random_sample(self, pc, n):
        idx = np.random.permutation(pc.shape[0])
        if idx.shape[0] < n:
            idx = np.concatenate([idx, np.random.randint(pc.shape[0], size=n-pc.shape[0])])
        return pc[idx[:n]]
