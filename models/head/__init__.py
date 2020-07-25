from .bn_head import BNHead
from .reduction_head import ReductionHead

def build_head(name, in_features, out_features, reduction_rate=4, bias_freeze=False, bn_where='after'):
    assert in_features % reduction_rate == 0

    if name == 'BNHead':
        return BNHead(in_features, out_features, bias_freeze, bn_where)
    elif name == 'ReductionHead':
        return ReductionHead(in_features, in_features//reduction_rate, out_features, bias_freeze, bn_where)
    else:
        raise KeyError('config[model][head] must in [BNHead, ReductionHead]')