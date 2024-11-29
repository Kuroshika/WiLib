import torch.nn as nn
import torch
import torchvision
from builder.builder import ModelRegistry
from .base_method import BaseMethod
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms.simclr_transform import SimCLRTransform

class SimCLR(BaseMethod):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resnet = torchvision.models.resnet18()
        self.criterion = NTXentLoss()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.optimizer = torch.optim.SGD(self.backbone.parameters(), lr=0.06)
        self.total_loss = 0
        self.transform = None
        # todo: 添加transform
        
    def train(self,x):
        self.backbone.train()
        self.backbone.to(self.device)
        x0 = x
        x1 = self.transform(x)
        z0 = self.backbone(x0).squeeze()
        z1 = self.backbone(x1).squeeze()
        loss = self.criterion(z0, z1)
        self.total_loss += loss.detach()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss
    
    def predict(self,x):
        self.backbone.to(self.device)
        return self.backbone(x)