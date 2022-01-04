import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

import torch
import torch.nn as nn

from utils import summary
from models.pooling import build_pooling
from models.head import build_head
from models.backbone import build_backbone
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
        pooling_size=1,
        head='BNHead',
        bn_where='after',
        batch_norm_bias=True,
        use_tqdm=True,
        is_inference=False
    ):
        
        super(Baseline, self).__init__()
        self.head_name = head
        self.num_classes = num_classes
        self.is_inference = is_inference
        
        self.backbone, feature_dim = build_backbone(backbone, pretrained=pretrained, progress=use_tqdm)
        self.global_pooling = build_pooling(pooling, pooling_size)
        self.head = build_head(head, feature_dim, self.num_classes, bias_freeze=not batch_norm_bias, bn_where=bn_where, pooling_size=pooling_size)

    def forward(self, x):
        x = self.backbone(x)
        # x.size = (batch_size, feature_dim, H, W)
        x = self.global_pooling(x)
        x = self.head(x)
        if self.is_inference:
            x = torch.sigmoid(x)
        return x
    
    def get_heat_maps_with_cam(self, x, return_output=True):
        r''' Get heatmaps using Class Activation Mapping: https://arxiv.org/pdf/1512.04150v1.pdf
        '''
        assert self.head_name == 'BNHead', 'Get heatmaps using Class Activation Mapping only work with BNHead'
        x = self.backbone(x)
        feat = x
        fc_weights = list(self.head.linear.parameters())[0].data
        fc_weights = fc_weights.view(1, self.num_classes, feat.size(1), 1, 1)
        # fc_weights.avgpoolsize() = (batch_size, num_classes, 2048, 1, 1)
        feat = feat.unsqueeze(dim=1)
        # feat.size() = (batch_size, 1, 2048, H, W)
        heatmaps = feat * fc_weights
        heatmaps = heatmaps.sum(dim=2)

        if return_output:
            x = self.global_pooling(x)
            x = self.head(x)
            return x, heatmaps
        return heatmaps

