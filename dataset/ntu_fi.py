import numpy as np
import glob
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
from builder.builder import DatasetRegistry


@DatasetRegistry.register_module()
class NTU_Fi(Dataset):
    """ CSI dataset.
    """

    def __init__(self, data_path, mode, modal='CSIamp', transform=None, few_shot=False, k=5, single_trace=True):
        """
        Args:
            data_path (string): Directory with all the images.
            modal (CSIamp/CSIphase): CSI data modal
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = data_path
        self.modal = modal
        self.transform = transform
        self.data_list = glob.glob(data_path + '/*/*.mat')
        self.folder = glob.glob(data_path + '/*/')
        self.category = {self.folder[i].split('/')[-2]: i for i in range(len(self.folder))}
        pass
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_dir = self.data_list[idx]
        y = self.category[sample_dir.split('/')[-2]]
        x = sio.loadmat(sample_dir)[self.modal]

        # normalize
        x = (x - 42.3199) / 4.9802

        # sampling: 2000 -> 500
        x = x[:, ::4]
        x = x.reshape(3, 114, 500)

        if self.transform:
            x = self.transform(x)

        x = torch.FloatTensor(x)

        return x, y


if __name__ == "__main__":
    dataset = NTU_Fi("/media/sda/datasets/NTU-Fi_HAR/test_amp")
    for d in dataset:
        print(type(d))