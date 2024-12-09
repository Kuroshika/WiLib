import torchvision
import numpy as np
import torch.nn as nn


class resnet_backbone(nn.Module):
    def __init__(self, model_name: str = "resnet18", pretrained=False):
        super().__init__()
        self.pretrained = pretrained
        self.model_name = model_name
        resnet = self._get_resnet_model()
        self.model = nn.Sequential(*list(resnet.children())[:-1])

    def _get_resnet_model(self):
        if self.model_name == "resnet18":
            return torchvision.models.resnet18(pretrained=self.pretrained)
        if self.model_name == "resnet34":
            return torchvision.models.resnet34(pretrained=self.pretrained)
        if self.model_name == "resnet50":
            return torchvision.models.resnet50(pretrained=self.pretrained)

    def forward(self, x):
        return self.model(x)
