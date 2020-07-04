import torch
import torch.nn as nn

from .cross_entropy_smooth import CrossEntropyLabelSmooth
from .hard_mine_triplet_loss import TripletLoss
from .center_loss import CenterLoss

class Softmax_Triplet_loss(nn.Module):
    def __init__(self, num_class, margin, epsilon, use_gpu):
        super().__init__()
        self.cross_entropy = CrossEntropyLabelSmooth(num_classes=num_class, epsilon=epsilon, use_gpu=use_gpu)
        self.triplet = TripletLoss(margin=margin)
        
    
    def forward(self, score, feat, target):
        return self.cross_entropy(score, target) + self.triplet(feat, target) 