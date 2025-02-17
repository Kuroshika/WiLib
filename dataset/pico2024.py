import random

import numpy as np
import glob
import scipy.io as sio
import torch
import os
from torch.utils.data import Dataset, DataLoader
from builder.builder import DatasetRegistry


@DatasetRegistry.register_module()
class Pico2024(Dataset):
    def __init__(self, data_path, label_path='', mode='train',**kwargs):
        self.mode = mode
        self.label_names = ['walking','running','falling','no_activity']
        self.data_path = data_path
        self.label_path = label_path
        self.load_data()
    
    def load_data(self):
        with open(self.label_path, 'r') as f:
            labels = f.readlines()
        self.samples = [i.split(",") for i in labels]
        self.data = np.load(self.data_path)


    def __getitem__(self, index):
        return self.data[index], self.label_names.index(self.samples[index][1])

    def __len__(self):
        return len(self.samples)

    def uniform_sample_np(self, data_numpy, size):
        T, S, C = data_numpy.shape
        if T == size:
            return data_numpy
        interval = T / size
        uniform_list = [int(i * interval) for i in range(size)]
        return data_numpy[ uniform_list,:,:]


    def random_sample_np(self,data_numpy, size):
        C, T, V, M = data_numpy.shape
        if T == size:
            return data_numpy
        interval = int(np.ceil(size / T))
        random_list = sorted(random.sample(list(range(T))*interval, size))
        return data_numpy[:, random_list]

        