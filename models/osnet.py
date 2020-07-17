import torch
import torchvision
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from torchsummary import summary

import sys
sys.path.append('.')

from utils import summary

class Standard3x3Conv(nn.Module):
    ''' Standard 3 × 3 convolution
    '''
    def __init__(self, in_channels, out_channels):
        super(Standard3x3Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Lite3x3Conv(nn.Module):
    ''' Lite 3 × 3 convolution: use pointwise -> depthwise instead of depthwise -> pointwise
    '''
    def __init__(self, in_channels, out_channels):
        super(Lite3x3Conv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Conv1x1(nn.Module):
    """1x1 convolution"""

    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class Conv1x1Linear(nn.Module):
    """1x1 convolution without relu"""

    def __init__(self, in_channels, out_channels):
        super(Conv1x1Linear, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class ChannelGate(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input."""

    def __init__(
        self,
        in_channels,
        num_gates=None,
        return_gates=False,
        gate_activation='sigmoid',
        reduction=16,
        layer_norm=False
    ):
        super(ChannelGate, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            in_channels,
            in_channels // reduction,
            kernel_size=1,
            bias=True,
            padding=0
        )
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm((in_channels // reduction, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            in_channels // reduction,
            num_gates,
            kernel_size=1,
            bias=True,
            padding=0
        )
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError(
                "Unknown gate activation: {}".format(gate_activation)
            )

    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input * x
    

class BaselineBottleneck(nn.Module):
    ''' Baseline bottleneck
    '''
    def __init__(self, in_channels, out_channels):
        super(BaselineBottleneck, self).__init__()
        mid_channels = out_channels // 4
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2 = Lite3x3Conv(mid_channels, mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)
    
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out = x + residual
        return F.relu(out)

class OSBlock(nn.Module):
    """ Omni-scale feature learning block
    """
    def __init__(self, in_channels, out_channels):
        super(OSBlock, self).__init__()
        mid_channels = out_channels // 4
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = Lite3x3Conv(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(
            Lite3x3Conv(mid_channels, mid_channels),
            Lite3x3Conv(mid_channels, mid_channels),
        )
        self.conv2c = nn.Sequential(
            Lite3x3Conv(mid_channels, mid_channels),
            Lite3x3Conv(mid_channels, mid_channels),
            Lite3x3Conv(mid_channels, mid_channels),
        )
        self.conv2d = nn.Sequential(
            Lite3x3Conv(mid_channels, mid_channels),
            Lite3x3Conv(mid_channels, mid_channels),
            Lite3x3Conv(mid_channels, mid_channels),
            Lite3x3Conv(mid_channels, mid_channels),
        )
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)
    
    def forward(self, x):
        residual = x
        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)
        x2 = self.gate(x2a) + self.gate(x2b) + self.gate(x2c) + self.gate(x2d)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out = x3 + residual
        return F.relu(out)

class OSNet(nn.Module):
    ''' https://arxiv.org/pdf/1905.00953.pdf
    '''
    def __init__(self, num_classes, channels=[64, 256, 384, 512], feature_dim=512, is_training=True):
        super(OSNet, self).__init__()
        self.is_training = is_training
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Sequential(
            OSBlock(channels[0], channels[1]),
            OSBlock(channels[1], channels[1]),
            Conv1x1(channels[1], channels[1]),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            OSBlock(channels[1], channels[2]),
            OSBlock(channels[2], channels[2]),
            Conv1x1(channels[2], channels[2]),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.conv4 = nn.Sequential(
            OSBlock(channels[2], channels[3]),
            OSBlock(channels[3], channels[3])
        )
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Sequential(
            nn.Linear(channels[3], feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

        self.init_params()

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.classifier(x)
        if not self.is_training:
            x = torch.sigmoid(x)
        return x

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

if __name__ == "__main__":
    model = OSNet(num_classes=26)
    summary(model, input_size=(3, 256, 128), batch_size=32, device='cpu')
    params = model.state_dict()
    pass