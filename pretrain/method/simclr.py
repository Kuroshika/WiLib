import torch.nn as nn
import torch
import torchvision
from builder.builder import ModelRegistry
from .base_method import BaseMethod
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms.simclr_transform import SimCLRTransform
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset.utils.Data_Augument import DataTransform


class SimCLR(BaseMethod):
    def __init__(self, **Aug_args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resnet = torchvision.models.resnet18()
        self.criterion = NTXentLoss()  # Use the Normalized Temperature-scaled Cross Entropy Loss.
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = SimCLRProjectionHead(input_dim=512, hidden_dim=512, output_dim=128, num_layers=3)
        self.optimizer = torch.optim.SGD(
            [
                {"params": self.backbone.parameters(), "lr": 0.06},
                {"params": self.projection_head.parameters(), "lr": 0.06},
            ],
            weight_decay=1e-5,
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=10, eta_min=0.01)
        self.total_loss = 0

        # todo: 添加transform
        self.augmentation = Aug_args

    def train(self, x):
        self.backbone.train()
        self.projection_head.train()
        self.backbone.to(self.device)
        self.projection_head.to(self.device)

        # This code is just test for the UT-HAR dataset
        x = x.view(-1, 250, 3, 30)  # (batch_size, 250, 3, 30)
        # The augument part
        x = x.view(-1, 250, 3 * 30)
        x = x.permute(0, 2, 1)
        x0 = DataTransform(x, self.augmentation, type="weak")
        x1 = DataTransform(x, self.augmentation, type="strong")

        x0 = self.convert_to_tensor(x0.reshape(-1, 3, 30, 250)).to(self.device)
        x1 = self.convert_to_tensor(x1.reshape(-1, 3, 30, 250)).to(self.device)
        x0 = x0.permute(0, 1, 3, 2)  # (batch_size, 3, 250, 30)
        x1 = x1.permute(0, 1, 3, 2)  # (batch_size, 3, 250, 30)

        z0 = self.backbone(x0).squeeze()
        z1 = self.backbone(x1).squeeze()
        z0 = self.projection_head(z0)
        z1 = self.projection_head(z1)

        loss = self.criterion(z0, z1)
        self.total_loss += loss.detach()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        return loss

    def predict(self, x):
        self.backbone.to(self.device)
        return self.backbone(x)

    def convert_to_tensor(self, data):
        if type(data) is torch.Tensor:
            return data.float()
        else:
            return torch.Tensor(data).float()
