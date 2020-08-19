import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))

import torch
import torch.nn as nn
import torch.nn.functional as F


from torchsummary import summary
from efficientnet_pytorch import EfficientNet

class Stem(nn.Module):
    def __init__(self, in_channels):
        pass


class Module1(nn.Module):
    def __init__(self, in_chanels, out_channels, activation):
        super(Module1, self).__init__()
        self.depthwise = nn.Conv2d(in_channels=in_chanels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = activation

class Module2(nn.Module):
    def __init__(self):
        pass


class Efficient(nn.Module):
    def __init__(self, name='efficientnet-b5', advprop=True):
        super(Efficient, self).__init__()
        self.model = EfficientNet.from_pretrained(name, advprop=advprop)
        
    def forward(self, x):
        return self.model.extract_features(x)
    
    def get_outchannels(self):
        return self.model._bn1.num_features

if __name__ == "__main__":
    model = Efficient('efficientnet-b1')
    summary(model, (3, 256, 128))
    model = nn.Linear()

