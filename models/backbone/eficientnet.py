import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F

from efficientnet_pytorch import EfficientNet

from utils import summary

class Efficient(nn.Module):
    def __init__(self, name='efficientnet-b5', advprop=True):
        super(Efficient, self).__init__()
        self.name = name
        self.model = EfficientNet.from_pretrained(name, advprop=advprop)
        
    def forward(self, x):
        return self.model.extract_features(x)
    
    def get_out_channels(self):
        return self.model._bn1.num_features
    
    def get_image_size(self):
        return EfficientNet.get_image_size(self.name)

if __name__ == "__main__":
    name = 'efficientnet-b5'
    model = EfficientNet.from_pretrained(name, advprop=True)
    print("Input image size: ", EfficientNet.get_image_size(name))
    # summary(model, input_data=(3, EfficientNet.get_image_size(name), EfficientNet.get_image_size(name)//2))
