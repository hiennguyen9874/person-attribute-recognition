import torch.optim as optim

def build_optimizers(config, param_groups):
    cfg_optimizer = config['optimizer']
    if cfg_optimizer['name'] == 'adam':
        dict_paramsters = {
            'lr': cfg_optimizer['lr'],
            'weight_decay': cfg_optimizer['weight_decay'],
            'beta1': cfg_optimizer['adam_beta1'],
            'beta2': cfg_optimizer['adam_beta2']
        }
        return optim.Adam(
            param_groups,
            lr=cfg_optimizer['lr'],
            weight_decay=cfg_optimizer['weight_decay'],
            betas=(cfg_optimizer['adam_beta1'],
                    cfg_optimizer['adam_beta2'])), dict_paramsters

    elif cfg_optimizer['name'] == 'sgd':
        dict_paramsters = {
            'lr': cfg_optimizer['lr'],
            'momentum': cfg_optimizer['momentum'],
            'weight_decay': cfg_optimizer['momentum'],
            'dampening': cfg_optimizer['sgd_dampening'],
            'nesterov': cfg_optimizer['sgd_nesterov']
        }
        return optim.SGD(
            param_groups,
            lr=cfg_optimizer['lr'],
            momentum=cfg_optimizer['momentum'],
            weight_decay=cfg_optimizer['weight_decay'],
            dampening=cfg_optimizer['sgd_dampening'],
            nesterov=cfg_optimizer['sgd_nesterov']), dict_paramsters
    else:
        raise KeyError('config[optimizer][name] error')
