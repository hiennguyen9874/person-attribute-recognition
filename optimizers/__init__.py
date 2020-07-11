import torch.optim as optim

def build_optimizers(config, model):
    cfg_optimizer = config['optimizer']
    specified_lr = False
    if 'specified_lr' in cfg_optimizer and len(cfg_optimizer['specified_lr']) > 0:
        specified_lr = True
        base_params = []
        new_params = []
        for name, module in model.named_children():
            if name not in cfg_optimizer['specified_lr']:
                base_params += [p for p in module.parameters()]
            else:
                new_params += [p for p in module.parameters()]
        param_groups = [{'params': base_params},
                        {'params': new_params, 'lr': cfg_optimizer['lr']}]
    else:
        param_groups = model.parameters()
        
    if cfg_optimizer['name'] == 'adam':
        dict_paramsters = {
            'lr': cfg_optimizer['lr']*0.1,
            'weight_decay': cfg_optimizer['weight_decay'],
            'beta1': cfg_optimizer['adam_beta1'],
            'beta2': cfg_optimizer['adam_beta2']
        }
        if specified_lr:
            dict_paramsters.update({"specified_lr": cfg_optimizer['specified_lr']})
        return optim.Adam(
            param_groups,
            lr=cfg_optimizer['lr']*0.1,
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
        if specified_lr:
            dict_paramsters.update({"specified_lr": cfg_optimizer['specified_lr']})
        return optim.SGD(
            param_groups,
            lr=cfg_optimizer['lr'],
            momentum=cfg_optimizer['momentum'],
            weight_decay=cfg_optimizer['weight_decay'],
            dampening=cfg_optimizer['sgd_dampening'],
            nesterov=cfg_optimizer['sgd_nesterov']), dict_paramsters
    else:
        raise KeyError('config[optimizer][name] error')
