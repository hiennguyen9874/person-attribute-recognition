from sys import maxunicode
import torch
from torch import matrix_power
import torchvision
import torch.nn as nn
from torch.nn import init

import sys
sys.path.append('.')

class VAC(nn.Module):
    ''' https://openaccess.thecvf.com/content_CVPR_2019/papers/Guo_Visual_Attention_Consistency_Under_Image_Transforms_for_Multi-Label_Image_Classification_CVPR_2019_paper.pdf
    CAM: https://arxiv.org/pdf/1512.04150.pdf
    '''
    __model_factory = {
        'resnet50': torchvision.models.resnet50,
        'resnet101': torchvision.models.resnet101
    }
    def __init__(self, num_classes, backbone='resnet50'):
        super(VAC, self).__init__()
        self.num_classes = num_classes
        
        resnet = self.__model_factory[backbone](pretrained=True)
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
        feat = x
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        # x.size() = (batch_size, 2048)
        x = self.classifier(x)

        fc_weights = list(self.classifier[0].parameters())[0].data
        fc_weights = fc_weights.view(1, self.num_classes, feat.size(1), 1, 1)
        # fc_weights.size() = (batch_size, num_classes, 2048, 1, 1)
        feat = feat.unsqueeze(dim=1)
        # feat.size() = (batch_size, 1, 2048, H, W)
        heatmaps = feat * fc_weights
        heatmaps = heatmaps.sum(dim=2)
        # heatmaps.size() = (2, 26, 8, 4)
        return x, heatmaps

if __name__ == "__main__":
    model = VAC(26)
    batch = torch.rand((2, 3, 256, 128))
    out, hm = model(batch)
    pass

