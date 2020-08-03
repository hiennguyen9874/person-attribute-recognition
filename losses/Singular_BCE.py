import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.nn.functional as F

class Singular_BCE(nn.Module):
    def __init__(self, reduction='mean'):
        super(Singular_BCE, self).__init__()
        self.reduction = reduction

    def forward(self, logits, targets, idx_attribute):
        return F.binary_cross_entropy_with_logits(logits.gather(1, idx_attribute.unsqueeze(1)).squeeze(), targets.gather(1, idx_attribute.unsqueeze(1)).squeeze(), reduction=self.reduction)