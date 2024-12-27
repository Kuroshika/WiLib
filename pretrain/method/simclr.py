import os
import random
from unittest import main

import torch
import torch.nn as nn
import torchvision
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms.simclr_transform import SimCLRTransform
from torch.optim.lr_scheduler import CosineAnnealingLR

from builder.builder import ModelRegistry
from dataset.utils.Data_Augument import DataTransform

from .base_method import BaseMethod


class SimCLR(BaseMethod):
    def __init__(self, backbone=None, projection_head=None, criterion=NTXentLoss(), linear_eval=False, **aug_args):
        # super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.augmentation = aug_args
        self.linear_eval = linear_eval

        # Initialize components and move them to the correct device.
        self._initialize_components(backbone, projection_head)

        # Set up the loss function.
        self.criterion = criterion

        # Configure optimizer based on whether we are in linear evaluation mode.
        self._configure_optimizer(self.linear_eval)

        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=10, eta_min=0.01)
        self.total_loss = 0

    def train_step(self, x):
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

    # TODO: Check the linear predict function, add it and the import the loss function to the linear eval engine.
    def linear_predict(self, x, mode=None):
        if not self.linear_eval:
            raise RuntimeError("Linear evaluation mode is not enabled.")
        self.backbone.eval()
        self.classifier.train() if mode == "train" else self.classifier.eval()
        with torch.no_grad():  # Disable gradient calculation during prediction.
            # This code is just test for the UT-HAR dataset
            x = x.view(-1, 250, 3, 30)  # (batch_size, 250, 3, 30)
            # The augument part
            x = x.view(-1, 250, 3 * 30)
            x = x.permute(0, 2, 1)
            x = DataTransform(x, self.augmentation, type=random.choice(["weak", "strong"]))

            x = self.convert_to_tensor(x.reshape(-1, 3, 30, 250)).to(device=self.device, dtype=torch.float32)
            x = x.permute(0, 1, 3, 2)  # (batch_size, 3, 250, 30)
            features = self.backbone(x.to(self.device)).squeeze()

        predictions = self.classifier(features)
        return predictions

    def convert_to_tensor(self, data):
        return (
            torch.tensor(data, dtype=torch.float32, device=self.device)
            if not isinstance(data, torch.Tensor)
            else data.to(self.device)
        )

    def _initialize_components(self, backbone, projection_head):
        """Initialize model components and move them to the correct device."""
        self.backbone = backbone.to(self.device) if backbone is not None else None
        self.projection_head = (
            projection_head.to(self.device)
            if projection_head is not None
            else SimCLRProjectionHead(input_dim=512, hidden_dim=512, output_dim=128).to(self.device)
        )

    def _configure_optimizer(self, linear_eval):
        """Configure the optimizer depending on whether linear evaluation is enabled."""
        if linear_eval:
            self.classifier = LinearClassifier(512, 7, self.device)
            for param in self.backbone.parameters():
                param.requires_grad = False
            params = [{"params": self.classifier.parameters(), "lr": 0.06}]
            self.optimizer = torch.optim.Adam(params, weight_decay=1e-5)
        else:
            params = [
                {"params": self.backbone.parameters(), "lr": 0.06},
                {"params": self.projection_head.parameters(), "lr": 0.06},
            ]
            self.optimizer = torch.optim.SGD(params, weight_decay=1e-5)


class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, device):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes).to(device)

    def forward(self, x):
        return self.fc(x)
