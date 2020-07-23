import torch
from .osnet import OSNet
from .baseline import Baseline
from .util import *

def build_model(config, num_classes, device=torch.device('cpu')):
    dict_paramsters = None
    if config['name'] == 'osnet':
        model = OSNet(num_classes=num_classes)

    elif config['name'] == 'baseline':
        dict_paramsters = {
            'backbone': config['backbone'],
            'pretrained': config['pretrained'],
            'pooling': config['pooling'],
            'batch_norm_bias': config['batch_norm_bias']}

        model = Baseline(
            num_classes=num_classes,
            backbone=config['backbone'],
            pretrained=config['pretrained'],
            pooling=config['pooling'],
            batch_norm_bias=config['batch_norm_bias'])

    elif config['name'] == 'osnet':
        dict_paramsters = {
            'pooling': config['pooling'],
            'batch_norm_bias': config['batch_norm_bias']}

        model = OSNet(
            num_classes=num_classes,
            pooling=config['pooling'],
            batch_norm_bias=config['batch_norm_bias'])

    else:
        raise KeyError('config[model][name] error')
    return model, dict_paramsters
