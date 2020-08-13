import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.nn.functional as F


class CEL_Sigmoid(nn.Module):
    r""" https://arxiv.org/pdf/2005.11909.pdf
    """
    def __init__(self, pos_ratio=None, reduction='mean', use_gpu=True):
        super(CEL_Sigmoid, self).__init__()
        assert reduction in ['sum', 'mean'], 'reduction must be mean or sum'
        self.pos_ratio = pos_ratio
        self.reduction = reduction
        self.use_gpu = use_gpu

    def forward(self, logits, targets):
        batch_size = logits.shape[0]
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        if self.pos_ratio is not None:
            weight = self.__ratio2weight(targets, self.pos_ratio)
            loss = (loss * weight)
        loss = loss.sum() / batch_size if self.reduction == 'mean' else loss.sum()
        return loss

    def __ratio2weight(self, targets, ratio):
        ratio = ratio.type_as(targets)
        weights = torch.exp(targets * (1 - ratio) + (1 - targets) * ratio)
        return weights