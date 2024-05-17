import random

import numpy as np
import glob
import scipy.io as sio
import torch
import os
from torch.utils.data import Dataset, DataLoader
from builder.builder import DatasetRegistry


@DatasetRegistry.register_module()
class UT_HAR(Dataset):
    def __init__(self, data_path, label_path='', mode='train', split_scale=0.8):
        self.mode = mode
        self.split_scale = split_scale
        self.data = None
        self.label = None
        self.num_classes = 7
        self.action_list = ['bed', 'fall', 'pickup', 'run', 'sitdown', 'standup', 'walk']
        assert type(data_path) == str and data_path != '', "data path should be a str type and not empty"
        self.data_path = data_path
        self.label_path = label_path
        self.load_data()

    def load_data(self):
        data_list = [os.path.join(self.data_path,f) for f in os.listdir(self.data_path)]
        label_list = [os.path.join(self.label_path,f) for f in os.listdir(self.label_path)]
        WiFi_data = {}
        for data_dir in data_list:
            data_name = data_dir.split('/')[-1].split('.')[0]
            with open(data_dir, 'rb') as f:
                data = np.load(f)
                data = data.reshape(len(data), 1, 250, 90)
                data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
            WiFi_data[data_name] = torch.Tensor(data_norm)
        for label_dir in label_list:
            label_name = label_dir.split('/')[-1].split('.')[0]
            with open(label_dir, 'rb') as f:
                label = np.load(f)
            WiFi_data[label_name] = torch.Tensor(label)

        if self.mode=='train':
            self.data = WiFi_data[f'X_{self.mode}']
            self.label = WiFi_data[f'y_{self.mode}']
        else:
            self.data = torch.cat((WiFi_data['X_val'],WiFi_data['X_test']),0)
            self.label = torch.cat((WiFi_data['y_val'],WiFi_data['y_test']),0)


    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)
