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
            for model_id in range(TRAIN_START, TRAIN_END+1):
                partial_paths.append(os.path.join(self.dataroot, 'partial', '{}.npy'.format(model_id)))
                complete_paths.append(os.path.join(self.dataroot, 'complete', '{}.xyz'.format(model_id)))
        elif self.split == 'valid':
            for model_id in range(VALID_START, VALID_END+1):
                partial_paths.append(os.path.join(self.dataroot, 'partial', '{}.npy'.format(model_id)))
                complete_paths.append(os.path.join(self.dataroot, 'complete', '{}.xyz'.format(model_id)))
        elif self.split == 'test':
            for model_id in range(TEST_START, TEST_END+1):
                partial_paths.append(os.path.join(self.dataroot, 'partial', '{}.npy'.format(model_id)))
                complete_paths.append(os.path.join(self.dataroot, 'complete', '{}.xyz'.format(model_id)))
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
