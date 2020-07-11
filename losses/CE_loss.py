import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import margin_ranking_loss

def ratio2weight(targets, ratio):
    ratio = ratio.type_as(targets)
    pos_weights = targets * (1 - ratio)
    neg_weights = (1 - targets) * ratio
    weights = torch.exp(neg_weights + pos_weights)

    # # for RAP dataloader, targets element may be 2, with or without smooth, some element must great than 1
    # weights[targets > 1] = 0.0

    return weights

class CEL_Sigmoid(nn.Module):
    ''' https://arxiv.org/pdf/2005.11909.pdf
    '''
    def __init__(self, pos_ratio=None, reduction='mean', use_gpu=True):
        super(CEL_Sigmoid, self).__init__()
        assert reduction in ['sum', 'mean']
        self.pos_ratio = pos_ratio
        self.reduction = reduction
        self.use_gpu = use_gpu

    def forward(self, logits, targets):
        batch_size = logits.shape[0]
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        if self.pos_ratio is not None:
            weight = ratio2weight(targets, self.pos_ratio)
            loss = (loss * weight)
        loss = loss.sum() / batch_size if self.reduction == 'mean' else loss.sum()
        return loss

if __name__ == "__main__":
    target = torch.ones([10, 64], dtype=torch.float32).cuda() # 64 classes, batch size = 10
    output = torch.full([10, 64], 1.5).cuda() # A prediction (logit)
    pos_weight = torch.ones([64]).cuda() # All weights are equal to 1
    criterion1 = CEL_Sigmoid(pos_weight)
    temp = torch.ones(1).cuda()
    out = criterion1(output, target)
    pass