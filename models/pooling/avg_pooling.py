import sys
sys.path.append('.')

import torch
import torch.nn.functional as F
from torch import nn

class FastGlobalAvgPool2d(nn.Module):
    r""" copy from here: https://arxiv.org/pdf/2003.13630.pdf
    Args:
        flatten (boolean): view tensor from (batch_size, feature_dim, 1, 1) to (batch_size, feature_dim)
    """
    def __init__(self, flatten=False):
        super(FastGlobalAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)

if __name__ == "__main__":
    input = torch.rand((32, 2048, 16, 8))
    pooling = FastGlobalAvgPool2d(flatten=False)
    out = pooling(input)
    out2 = nn.AdaptiveAvgPool2d(1)(input)
    pass