import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))

import torch
import torch.nn.functional as F
from torch import nn

class FastGlobalAvgPool2d(nn.Module):
    r""" copy from here: https://arxiv.org/pdf/2003.13630.pdf
    Args:
        flatten (boolean): view tensor from (batch_size, feature_dim, 1, 1) to (batch_size, feature_dim)
    """
    def __init__(self, flatten=False, **kwargs):
        super(FastGlobalAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)

class AvgPooling2d(nn.Module):
    def __init__(self, output_size=1, **kwargs):
        super(AvgPooling2d, self).__init__()
        self.pooling = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        x = self.pooling(x)
        return x
