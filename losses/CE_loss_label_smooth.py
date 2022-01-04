import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.jit.script
def _ratio2weight(targets, ratio):
    ratio = ratio.type_as(targets)
    weights = torch.exp(targets * (1 - ratio) + (1 - targets) * ratio)
    return weights

class CEL_Sigmoid_Smooth(nn.Module):
    r""" https://arxiv.org/pdf/2005.11909.pdf
    """
    def __init__(self, num_classes, epsilon=0.1, pos_ratio=None, reduction='mean', **kwargs):
        super(CEL_Sigmoid_Smooth, self).__init__()
        assert reduction in ['sum', 'mean'], 'reduction must be mean or sum'
        self.pos_ratio = pos_ratio
        self.reduction = reduction
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, inputs, targets):
        # batch_size = inputs.shape[0]
        smoothed = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = F.binary_cross_entropy_with_logits(inputs, smoothed, reduction='none')
        if self.pos_ratio is not None:
            weight = _ratio2weight(targets, self.pos_ratio)
            loss = (loss * weight)
        # loss = loss.sum() / batch_size if self.reduction == 'mean' else loss.sum()
        loss = loss.mean(dim=0)
        return loss.mean() if self.reduction == 'mean' else loss.sum()
