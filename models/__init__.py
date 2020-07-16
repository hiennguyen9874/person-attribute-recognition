import torch
from .osnet import OSNet
from .baseline_reid import BaselineReid
from .baseline_attribute import BaselineAttribute
from .alm import BaselineALM

__model_factory = {
    'osnet': OSNet,
    'baseline_reid': BaselineReid,
    'baseline_attribute': BaselineAttribute,
    'baseline_alm': BaselineALM
}

def build_model(config, num_classes, pretrained=True, device=torch.device('cpu')):
    # if config['name'] not in list(__model_factory.keys()):
    #     raise KeyError
    # return __model_factory[config['name']](num_classes=num_classes)
    dict_paramsters = None
    if config['name'] == 'osnet':
        model =  OSNet(num_classes=num_classes)
    elif config['name'] == 'baseline_reid':
        dict_paramsters = {'backbone': config['backbone']}
        model = BaselineReid(num_classes=num_classes, backbone=config['backbone'], pretrained=pretrained)
    elif config['name'] == 'baseline_attribute':
        dict_paramsters = {'backbone': config['backbone']}
        model = BaselineAttribute(num_classes=num_classes, backbone=config['backbone'], pretrained=pretrained)
    elif config['name'] == 'baseline_alm':
        dict_paramsters = {'backbone': config['backbone']}
        model = BaselineALM(num_classes=num_classes, backbone=config['backbone'], pretrained=pretrained, device=device)
    else:
        raise KeyError('config[model][name] error')
    return model, dict_paramsters