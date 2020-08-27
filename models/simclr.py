import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

import torch
import torch.nn as nn

from torchsummary import summary

from models.backbone import build_backbone
from models.pooling import build_pooling

class SimCLR(nn.Module):
    def __init__(
        self, 
        out_dim, 
        backbone='resnet50', 
        pretrained=True, 
        progress=True,
        pooling='avg_pooling',
        pooling_size=1,
        ):
        super(SimCLR, self).__init__()
        self.backbone, feature_dim = build_backbone(backbone, pretrained, progress)
        self.global_pooling = build_pooling(pooling, pooling_size)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, out_dim)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.global_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x

if __name__ == "__main__":
    model = SimCLR(256)
    summary(model, (3, 224, 224))

