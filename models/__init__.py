from .osnet import OSNet
from .baseline_reid import BaselineReid
from .baseline_attribute import BaselineAttribute

__model_factory = {
    'osnet': OSNet,
    'baseline_reid': BaselineReid,
    'baseline_attribute': BaselineAttribute
}

def build_model(config, num_classes, pretrained=True, use_gpu=True):
    if config['name'] not in list(__model_factory.keys()):
        raise KeyError
    return __model_factory[config['name']](num_classes=num_classes)