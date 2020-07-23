import torch.nn as nn
import sys
sys.path.append('.')

from models.util import get_norm

class ReductionHead(nn.Module):
    def __init__(self, in_features, hidden_feature, out_features, bias_freeze):
        self.bottleneck = nn.Sequential(
            nn.Linear(in_features, hidden_feature),
            get_norm(hidden_feature, type_norm='2d', bias_freeze=bias_freeze),
            nn.LeakyReLU(p=0.2),
            nn.Dropout()
        )
        self.bnneck = get_norm(hidden_feature, '2d',  bias_freeze)
        self.linear = nn.Linear(hidden_feature, out_features)
    
    def forward(self, x):
        x = self.bottleneck(x)
        x = self.bnneck(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
    
