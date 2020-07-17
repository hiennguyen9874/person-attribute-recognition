import torch
from torch.nn.modules.batchnorm import BatchNorm2d
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('.')

from utils import summary

class SAM(nn.Module):
    """ Spatial attention module (https://arxiv.org/pdf/1910.04562.pdf)
    """ 
    def __init__(self, in_channels, out_channels):
        super(SAM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        out = torch.mean(x, dim=1, keepdim=True)
        out = self.conv(out)
        out = x * out
        out = self.avg(out)
        out = out.view(x.size(0), -1)
        out = self.fc(out)
        return out

class BaselineSAM(nn.Module):
    __model_factory = {
        'resnet50': torchvision.models.resnet50,
        'resnet101': torchvision.models.resnet101
    }
    def __init__(self, num_classes, backbone='resnet50', last_stride_1=True, pretrained=True):
        super(BaselineSAM, self).__init__()
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

        self.classifier = nn.ModuleList()
        for _ in range(num_classes):
            self.classifier.append(SAM(2048, 1))
        # TODO: Add batch norm and test it
        self.bn = nn.BatchNorm1d(self.num_classes)

    def forward(self, x):
        x = self.base(x)
        # x.size = (batch_size, 2048, 16, 8)
        out = []
        for i in range(self.num_classes):
            out.append(self.classifier[i](x))
        return self.bn(torch.cat(out, dim=1))

if __name__ == "__main__":
    model = BaselineSAM(26)
    # batch = torch.rand((4, 3, 256, 128))
    # out = model(batch)
    summary(print, model, (3, 256, 128), 64, 'cpu', False)
    pass