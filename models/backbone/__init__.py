import sys
sys.path.append('.')

from models.backbone.resnet import *
from models.backbone.resnet_nl import *
from models.backbone.resnet_ibn_a import *
from models.backbone.resnet_ibn_a_nl import *
from models.backbone.osnet import *

__backbones = {
    'osnet': (osnet, 512),
    'resnet50': (resnet50, 2048),
    'resnet101': (resnet101, 2048),
    'resnet50_nl': (resnet50_nl, 2048),
    'resnet101_nl': (resnet101_nl, 2048),
    'resnet50_ibn_a': (resnet50_ibn_a, 2048),
    'resnet101_ibn_a': (resnet101_ibn_a, 2048),
    'resnet50_ibn_a_nl': (resnet50_ibn_a_nl, 2048),
    'resnet101_ibn_a_nl': (resnet101_ibn_a_nl, 2048),
}

def build_backbone(name, pretrained=True, progress=True):
    assert name in __backbones.keys(), 'name of backbone must in %s' % str(__backbones.keys())
    return __backbones[name][0](pretrained=pretrained, progress=progress), __backbones[name][1]

if __name__ == "__main__":
    build_backbone('afdsa')