import torch
from .baseline import Baseline
from .util import *

def build_model(config, num_classes, device=torch.device('cpu')):
    cfg_model = config['model']
    dict_paramsters = None
    
    if cfg_model['name'] == 'baseline':
        dict_paramsters = {
            'backbone': cfg_model['backbone'],
            'pretrained': cfg_model['pretrained'],
            'pooling': cfg_model['pooling'],
            'batch_norm_bias': cfg_model['batch_norm_bias'],
            'head': cfg_model['head'],
            'bn_where': cfg_model['bn_where']}

        model = Baseline(
            num_classes=num_classes,
            backbone=cfg_model['backbone'],
            pretrained=cfg_model['pretrained'],
            pooling=cfg_model['pooling'],
            head=cfg_model['head'],
            bn_where=cfg_model['bn_where'],
            batch_norm_bias=cfg_model['batch_norm_bias'],
            use_tqdm=cfg_model['use_tqdm'])

    else:
        raise KeyError('config[model][name] error')
    return model, dict_paramsters
