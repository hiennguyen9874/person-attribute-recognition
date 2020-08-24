import torch.nn as nn

__all__ = ['get_norm']

def get_norm(in_features, type_norm='2d', bias_freeze=False):
    assert type_norm in ['1d', '2d'], 'type_norm must be 1d or 2d'
    if type_norm == '1d':
        norm = nn.BatchNorm1d(in_features)
    elif type_norm == '2d':
        norm = nn.BatchNorm2d(in_features)
    if bias_freeze:
        norm.bias.requires_grad_(False)
    return norm

