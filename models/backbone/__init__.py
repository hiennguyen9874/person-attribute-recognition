import torch
import torchvision
import torch.nn as nn

def build_backbone(name, pretrained=True):
    if name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=pretrained)
        return nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4
        )
    elif name == 'resnet101':
        model = torchvision.models.resnet101(pretrained=pretrained)
        return nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4
        )
    elif name == 'resnet50_ibn_a':
        model = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=pretrained)
        return nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4
        )
    elif name == 'resnet101_ibn_a':
        model = torch.hub.load('XingangPan/IBN-Net', 'resnet101_ibn_a', pretrained=pretrained)
        return nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4
        )
    else:
        raise KeyError('name backbone error, name must in [resnet50, resnet101, resnet50_ibn_a, resnet101_ibn_a]')