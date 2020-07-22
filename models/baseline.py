import torch
from torch import batch_norm
import torchvision
import torch.nn as nn
from torch.nn import init

import sys
sys.path.append('.')

from utils import summary
from models.pooling import build_pooling
from models.backbone import build_backbone
from models.weight_init import weights_init_classifier, weights_init_kaiming

class Baseline(nn.Module):
    ''' Model inspired https://arxiv.org/pdf/2005.11909.pdf
        Using grad-CAM to get heatmaps, inspired from https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
    '''
    def __init__(self, num_classes, backbone='resnet50', pretrained=True, pooling='avg_pooling', batch_norm_bias=True):
        super(Baseline, self).__init__()
        self.num_classes = num_classes
        
        self.backbone = build_backbone(backbone, pretrained=pretrained)

        self.avgpool = build_pooling(pooling)
        self.linear = nn.Linear(2048, self.num_classes)
        self.bn = nn.BatchNorm1d(self.num_classes)
        
        # freeze bias of batch_norm layer
        if not batch_norm_bias:
            self.bn.bias.requires_grad_(False)

        self.linear.apply(weights_init_classifier)
        self.bn.apply(weights_init_kaiming)

        # # heatmaps
        # self.gradients = None

    def forward(self, x):
        x = self.backbone(x)
        # x.size = (batch_size, 2048, 16, 8)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        # x.size() = (batch_size, 2048)
        x = self.linear(x)
        x = self.bn(x)
        return x
    
    def get_heat_maps_with_cam(self, x, return_output=True):
        r''' Get heatmaps using Class Activation Mapping: https://arxiv.org/pdf/1512.04150v1.pdf
        '''
        x = self.backbone(x)
        feat = x
        fc_weights = list(self.linear.parameters())[0].data
        fc_weights = fc_weights.view(1, self.num_classes, feat.size(1), 1, 1)
        # fc_weights.size() = (batch_size, num_classes, 2048, 1, 1)
        feat = feat.unsqueeze(dim=1)
        # feat.size() = (batch_size, 1, 2048, H, W)
        heatmaps = feat * fc_weights
        heatmaps = heatmaps.sum(dim=2)

        if return_output:
            x = self.avgpool(x)
            x = x.view(x.shape[0], -1)
            x = self.linear(x)
            x = self.bn(x)
            return x, heatmaps
        return heatmaps
    
    # # hook for the gradients of the activations
    # def activations_hook(self, grad):
    #     self.gradients = grad

    # def get_heat_maps_with_grad_cam(self, x):
    #     x = self.backbone(x)

    #     # register the hook
    #     h = x.register_hook(self.activations_hook)

    #     # apply the remaining pooling
    #     x = self.avgpool(x)
    #     x = x.view(x.shape[0], -1)
    #     x = self.linear(x)
    #     x = self.bn(x)
    #     return x
    
    # # method for the gradient extraction
    # def get_activations_gradient(self):
    #     return self.gradients
    
    # # method for the activation exctraction
    # def get_activations(self, x):
    #     return self.backbone(x)

if __name__ == "__main__":
    model = Baseline(26, 'resnet50', True, 'gem_pooling', True)
    summary(print, model, (3, 256, 128), 64, 'cpu', False)