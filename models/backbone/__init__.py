from .resnet import *
from .resnet_nl import *
from .resnet_ibn_a import *
from .resnet_ibn_a_nl import *

__backbones = {
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet50_nl': resnet50_nl,
    'resnet101_nl': resnet101_nl,
    'resnet50_ibn_a': resnet50_ibn_a,
    'resnet101_ibn_a': resnet101_ibn_a,
    'resnet50_ibn_a_nl': resnet50_ibn_a_nl,
    'resnet101_ibn_a_nl': resnet101_ibn_a_nl,
}

def build_backbone(name, pretrained=True):
    assert name in __backbones.keys()
    return __backbones[name](pretrained=pretrained)