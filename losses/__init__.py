import torch.nn as nn

def build_losses(config, weight):
    cfg_loss = config['loss']
    if cfg_loss['name'] == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss(weight=weight)
    else:
        raise KeyError('config[loss] error')