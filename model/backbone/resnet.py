import torchvision
import numpy as np
import torch.nn as nn


class resnet_backbone(nn.Module):
    def __init__(self, model_name: str = "resnet18", weights=None):
        super().__init__()
        self.weights = weights
        self.model_name = model_name
        resnet = self._get_resnet_model()
        self.model = nn.Sequential(*list(resnet.children())[:-1])

    def _get_resnet_model(self):
        if self.model_name == "resnet18":
            return torchvision.models.resnet18(weights=self.weights)
        if self.model_name == "resnet34":
            return torchvision.models.resnet34(weights=self.weights)
        if self.model_name == "resnet50":
            return torchvision.models.resnet50(weights=self.weights)

    def forward(self, x):
        return self.model(x)
