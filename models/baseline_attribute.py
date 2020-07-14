import torch
import torchvision
import torch.nn as nn
from torch.nn import init

import sys
sys.path.append('.')

class BaselineAttribute(nn.Module):
    ''' https://arxiv.org/pdf/2005.11909.pdf
    '''
    __model_factory = {
        'resnet50': torchvision.models.resnet50,
        'resnet101': torchvision.models.resnet101
    }
    def __init__(self, num_classes, baskbone='resnet50'):
        super(BaselineAttribute, self).__init__()
        self.num_classes = num_classes
        self.base = self.__model_factory[baskbone](pretrained=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(2048, self.num_classes),
            nn.BatchNorm1d(self.num_classes)
        )

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)

        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        # x.size = (batch_size, 2048, 16, 8)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        # x.size() = (batch_size, 2048)
        x = self.classifier(x)
        return x

