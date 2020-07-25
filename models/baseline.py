import torch
import torch.nn as nn

import sys
sys.path.append('.')

from torch.nn import init

from utils import summary
from models.pooling import build_pooling
from models.head import build_head
from models.backbone import build_backbone
from models.weight_init import weights_init_classifier, weights_init_kaiming
from models.util import *

class Baseline(nn.Module):
    r''' Model inspired https://arxiv.org/pdf/2005.11909.pdf
    '''
    def __init__(
        self,
        num_classes,
        backbone='resnet50',
        pretrained=True,
        pooling='avg_pooling',
        head='BNHead',
        bn_where='after',
        batch_norm_bias=True,
        use_tqdm=True):
        super(Baseline, self).__init__()
        self.num_classes = num_classes
        
        self.backbone = build_backbone(backbone, pretrained=pretrained, progress=use_tqdm)
        self.avgpool = build_pooling(pooling)
        self.head = build_head(head, 2048, self.num_classes, bias_freeze=not batch_norm_bias, bn_where=bn_where)

    def forward(self, x):
        x = self.backbone(x)
        # x.size = (batch_size, 2048, 16, 8)
        x = self.avgpool(x)
        x = self.head(x)
        return x
    
    def get_heat_maps_with_cam(self, x, return_output=True):
        r''' Get heatmaps using Class Activation Mapping: https://arxiv.org/pdf/1512.04150v1.pdf
        '''
        x = self.backbone(x)
        feat = x
        fc_weights = list(self.linear.parameters())[0].data
        fc_weights = fc_weights.view(1, self.num_classes, feat.size(1), 1, 1)
        # fc_weights.avgpoolsize() = (batch_size, num_classes, 2048, 1, 1)
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
    model = Baseline(26, 'resnet50_ibn_a_nl', True, 'gem_pooling', True)
    summary(print, model, (3, 256, 128), 64, 'cpu', False)

