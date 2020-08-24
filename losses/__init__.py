import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))

import torch
import torch.nn as nn

from losses.CE_loss import CEL_Sigmoid
from losses.Singular_BCE import Singular_BCE
from losses.Focal_loss import FocalLoss

def build_losses(config, pos_ratio, num_attribute, use_gpu=True, **kwargs):
    cfg_loss = config['loss']
    if cfg_loss['name'] == 'BCEWithLogitsLoss':
        pos_weight = torch.exp(-1 * pos_ratio)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight), {}
    elif cfg_loss['name'] == 'Non_BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss(), {}
    elif cfg_loss['name'] == 'CEL_Sigmoid':
        return CEL_Sigmoid(pos_ratio, use_gpu=use_gpu, reduction=cfg_loss['reduction']), {'reduction': cfg_loss['reduction']}
    elif cfg_loss['name'] == 'Singular_BCE':
        return Singular_BCE(num_attribute, reduction=cfg_loss['reduction']), {'reduction': cfg_loss['reduction']}
    elif cfg_loss['name'] == 'FocalLoss':
        return FocalLoss(pos_ratio), {}
    else:
        raise KeyError('config[loss][name] error')


