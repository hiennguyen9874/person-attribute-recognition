import torch.nn as nn
import sys
sys.path.append('.')

from models.util import get_norm

class BNHead(nn.Module):
    def __init__(self, in_features, out_features, bias_freeze):
        self.bnneck = get_norm(in_features, '2d',  bias_freeze)
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        x = self.bnneck(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
    
