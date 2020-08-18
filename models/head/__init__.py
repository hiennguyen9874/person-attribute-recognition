import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))

from models.head.bn_head import BNHead
from models.head.reduction_head import ReductionHead

def build_head(name, in_features, out_features, reduction_ratio=4, bias_freeze=False, bn_where='after', pooling_size=1):
    assert in_features % reduction_ratio == 0, 'in_channel must divide by reduction_ratio'

    if name == 'BNHead':
        return BNHead(in_features, out_features, bias_freeze, bn_where, pooling_size)
    elif name == 'ReductionHead':
        return ReductionHead(in_features, in_features//reduction_ratio, out_features, bias_freeze, bn_where, pooling_size)
    else:
        raise KeyError('config[model][head] must in [BNHead, ReductionHead]')

