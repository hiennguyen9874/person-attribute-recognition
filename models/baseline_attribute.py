import torch
import torchvision
import torch.nn as nn
from torch.nn import init

import sys
sys.path.append('.')

from utils import summary

class BaselineAttribute(nn.Module):
    ''' https://arxiv.org/pdf/2005.11909.pdf
    '''
    __model_factory = {
        'resnet50': torchvision.models.resnet50,
        'resnet101': torchvision.models.resnet101
    }
    def __init__(self, num_classes, backbone='resnet50', last_stride_1=True, pretrained=True):
        super(BaselineAttribute, self).__init__()
        self.num_classes = num_classes
        
        resnet = self.__model_factory[backbone](pretrained=pretrained)
        if last_stride_1:
            # remove the final downsample of resnet
            resnet.layer4[0].downsample[0].stride = (1, 1)
            resnet.layer4[0].conv2.stride=(1,1)

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

        self.classifier = nn.Sequential(
            nn.Linear(2048, self.num_classes),
            nn.BatchNorm1d(self.num_classes)
        )

    def forward(self, x):
        x = self.base(x)
        # x.size = (batch_size, 2048, 16, 8)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        # x.size() = (batch_size, 2048)
        x = self.classifier(x)
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
            x = self.classifier(x)
            return x, heatmaps
        return heatmaps

if __name__ == "__main__":
    model = BaselineAttribute(26)
    summary(print, model, (3, 256, 128), 64, 'cpu', True)