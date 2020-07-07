import torch.optim as optim
from .scheduler import WarmupMultiStepLR

def build_optimizers(config, param_groups):
    cfg_optimizer = config['optimizer']
    if cfg_optimizer['name'] == 'adam':
        return optim.Adam(
            param_groups,
            lr=cfg_optimizer['lr'],
            weight_decay=cfg_optimizer['weight_decay'],
            betas=(cfg_optimizer['adam_beta1'], cfg_optimizer['adam_beta2']))

    elif cfg_optimizer['name'] == 'sgd':
        return optim.SGD(
            param_groups,
            lr=cfg_optimizer['lr'],
            momentum=cfg_optimizer['momentum'],
            weight_decay=cfg_optimizer['weight_decay'],
            dampening=cfg_optimizer['sgd_dampening'],
            nesterov=cfg_optimizer['sgd_nesterov'])
    else:
        raise KeyError('config[optimizer][name] error')

def build_lr_scheduler(config, optimizer):
    cfg_lr_scheduler = config['lr_scheduler']
    if cfg_lr_scheduler['name'] == "warmup":
        return WarmupMultiStepLR(
            optimizer,
            milestones=cfg_lr_scheduler['steps'],
            gamma=cfg_lr_scheduler['gamma'],
            warmup_factor=cfg_lr_scheduler['factor'],
            warmup_iters=cfg_lr_scheduler['iters'],
            warmup_method=cfg_lr_scheduler['method'])
    else:
        raise KeyError('config[lr_scheduler][name] error')