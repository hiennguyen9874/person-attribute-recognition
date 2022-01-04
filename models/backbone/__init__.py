import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))

from models.backbone.resnet import *
from models.backbone.resnet_nl import *
from models.backbone.resnet_ibn_a import *
from models.backbone.resnet_ibn_a_nl import *
from models.backbone.osnet import *
from models.backbone.vgg import *
from models.backbone.eficientnet import Efficient
from models.backbone.timm_extractor import TimmModel

__backbones = {
    "osnet": (osnet, 512),
    "resnet50": (resnet50, 2048),
    "resnet101": (resnet101, 2048),
    "resnet50_nl": (resnet50_nl, 2048),
    "resnet101_nl": (resnet101_nl, 2048),
    "resnet50_ibn_a": (resnet50_ibn_a, 2048),
    "resnet101_ibn_a": (resnet101_ibn_a, 2048),
    "resnet50_ibn_a_nl": (resnet50_ibn_a_nl, 2048),
    "resnet101_ibn_a_nl": (resnet101_ibn_a_nl, 2048),
    "vgg16": (vgg16, 512),
    "vgg19": (vgg19, 512),
    "vgg16_bn": (vgg16_bn, 512),
    "vgg19_bn": (vgg19_bn, 512),
}


def build_backbone(name, pretrained=True, progress=True):
    # assert name in __backbones.keys(), 'name of backbone must in %s' % str(__backbones.keys())
    # return __backbones[name][0](pretrained=pretrained, progress=progress), __backbones[name][1]
    if name in __backbones.keys():
        return (
            __backbones[name][0](pretrained=pretrained, progress=progress),
            __backbones[name][1],
        )
    elif name.split("/")[0] == "timm":
        model_name = name.split("/")[1]
        model = TimmModel(model_name, pretrained=pretrained)
        return model, model.get_feature_dim()
    elif name in ["efficientnet-b" + str(i) for i in range(9)]:
        model = Efficient(name)
        return model, model.get_out_channels()


if __name__ == "__main__":
    import torch
    from utils import summary

    model, feature_dim = build_backbone("tim/tf_efficientnet_b2_ns")
    result = summary(
        model, torch.zeros(64, 3, 260, 130), batch_dim=None, print_step=True
    )
