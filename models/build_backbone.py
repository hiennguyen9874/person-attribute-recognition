import torch
import torchvision

def build_backbone(name, pretrained=True, last_stride_1=True):
    if name == 'resnet18':
        model =  torchvision.models.resnet50(pretrained=pretrained)
    elif name == 'resnet34':
        model = torchvision.models.resnet34(pretrained=pretrained)
    elif name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=pretrained)
    elif name == 'resnet101':
        model = torchvision.models.resnet101(pretrained=pretrained)
    elif name == 'resnet50_ibn_a':
        model = torch.hub.load('XingangPan/IBN-Net', 'resnet50_ibn_a', pretrained=pretrained)
    else:
        raise KeyError('name backbone error')
    if last_stride_1:
        # remove the final downsample of resnet
        model.layer4[0].downsample[0].stride = (1, 1)
        model.layer4[0].conv2.stride=(1,1)
    return model