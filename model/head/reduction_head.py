import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))

import torch.nn as nn

from model.util import get_norm
from model.weight_init import weights_init_classifier, weights_init_kaiming

class ReductionHead(nn.Module):
    def __init__(self, in_features, hidden_feature, out_features, bias_freeze, bn_where='after', pooling_size=1):
        assert bn_where in ['before', 'after'], 'bn_where must be before or after'
        super(ReductionHead, self).__init__()
        self.bn_where = bn_where
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_features, hidden_feature, kernel_size=1, stride=1, bias=False),
            get_norm(hidden_feature, type_norm='2d', bias_freeze=bias_freeze),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout()
        )
        if bn_where == 'before':
            self.bnneck = get_norm(hidden_feature, '2d',  bias_freeze)
        else:
            self.bnneck = get_norm(out_features, '1d',  bias_freeze)
        self.linear = nn.Linear(hidden_feature*pooling_size*pooling_size, out_features)

        self.linear.apply(weights_init_classifier)
        self.bottleneck.apply(weights_init_kaiming)
        self.bnneck.apply(weights_init_kaiming)

    def forward(self, x):
        x = self.bottleneck(x)
        if self.bn_where == 'before':
            x = self.bnneck(x)
            x = x.view(x.size(0), -1)
            x = self.linear(x)
        elif self.bn_where == 'after':
            x = x.view(x.size(0), -1)
            x = self.linear(x)
            x = self.bnneck(x)
        return x
    
