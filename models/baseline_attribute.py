import torch
import torchvision
import torch.nn as nn
from torch.nn import init

import sys
sys.path.append('.')

from utils import summary
from .layer import GeneralizedMeanPoolingP
from .build_backbone import build_backbone
from .weight_init import weights_init_classifier, weights_init_kaiming

class BaselineAttribute(nn.Module):
    ''' https://arxiv.org/pdf/2005.11909.pdf
    '''
    def __init__(self, num_classes, backbone='resnet50', last_stride_1=True, pretrained=True):
        super(BaselineAttribute, self).__init__()
        self.num_classes = num_classes
        
        resnet = build_backbone(backbone, pretrained, last_stride_1)
        self.base = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.linear = nn.Linear(2048, self.num_classes)
        self.bn = nn.BatchNorm1d(self.num_classes)

        self.linear.apply(weights_init_classifier)
        self.bn.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.base(x)
        # x.size = (batch_size, 2048, 16, 8)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        # x.size() = (batch_size, 2048)
        x = self.linear(x)
        x = self.bn(x)
        return x
    
    def get_heat_maps(self, x, return_output=True):
        x = self.base(x)
        feat = x
        fc_weights = list(self.classifier[0].parameters())[0].data
        fc_weights = fc_weights.view(1, self.num_classes, feat.size(1), 1, 1)
        # fc_weights.size() = (batch_size, num_classes, 2048, 1, 1)
        feat = feat.unsqueeze(dim=1)
        # feat.size() = (batch_size, 1, 2048, H, W)
        heatmaps = feat * fc_weights
        heatmaps = heatmaps.sum(dim=2)

        if return_output:
            x = self.avgpool(x)
            x = x.view(x.shape[0], -1)
            x = self.linear(x)
            x = self.bn(x)
            return x, heatmaps
        return heatmaps

if __name__ == "__main__":
    model = BaselineAttribute(26)
    summary(print, model, (3, 256, 128), 64, 'cpu', True)