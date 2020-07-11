import torch.optim.lr_scheduler as lr_scheduler
from .scheduler import WarmupMultiStepLR

def build_lr_scheduler(config, optimizer):
    cfg_lr_scheduler = config['lr_scheduler']
    if cfg_lr_scheduler['name'] == "WarmupMultiStepLR":
        dict_paramsters = {
            'milestones': cfg_lr_scheduler['steps'],
            'gamma': cfg_lr_scheduler['gamma'],
            'warmup_factor': cfg_lr_scheduler['factor'],
            'warmup_iters': cfg_lr_scheduler['iters'],
            'warmup_method': cfg_lr_scheduler['method']
        }
        return WarmupMultiStepLR(
            optimizer,
            milestones=cfg_lr_scheduler['steps'],
            gamma=cfg_lr_scheduler['gamma'],
            warmup_factor=cfg_lr_scheduler['factor'],
            warmup_iters=cfg_lr_scheduler['iters'],
            warmup_method=cfg_lr_scheduler['method']), dict_paramsters
            
    elif cfg_lr_scheduler['name'] == 'ReduceLROnPlateau':
        dict_paramsters = {
            'factor': cfg_lr_scheduler['factor'],
            'patience': cfg_lr_scheduler['patience'],
            'min_lr': cfg_lr_scheduler['min_lr']
        }
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=cfg_lr_scheduler['factor'],
            patience=cfg_lr_scheduler['patience'],
            min_lr=cfg_lr_scheduler['min_lr']), dict_paramsters
    
    elif cfg_lr_scheduler['name'] == 'MultiStepLR':
        dict_paramsters = {
            'milestones': cfg_lr_scheduler['steps'],
            'gamma': cfg_lr_scheduler['gamma']
        }
        return lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg_lr_scheduler['steps'],
            gamma=cfg_lr_scheduler['gamma']), dict_paramsters
    else:
        raise KeyError('config[lr_scheduler][name] error')