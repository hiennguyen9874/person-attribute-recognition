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
    def __init__(self, num_classes, baskbone='resnet50', pretrained=True):
        super(BaselineAttribute, self).__init__()
        self.num_classes = num_classes
        
        resnet = self.__model_factory[baskbone](pretrained=pretrained)
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

