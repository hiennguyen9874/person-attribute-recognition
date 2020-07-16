import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('.')

class ChannelAttr(nn.Module):
    def __init__(self, in_channels, reduction_rate=16):
        super(ChannelAttr, self).__init__()
        assert in_channels % reduction_rate == 0
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // reduction_rate, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=in_channels//reduction_rate, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x

class ALM(nn.Module):
    """ Attribute Localization Module (https://arxiv.org/pdf/1910.04562.pdf)
    """ 
    def __init__(self, in_channels, device):
        super(ALM, self).__init__()
        self.device = device

        self.channel_attr = ChannelAttr(in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Linear(in_channels, 4)
        self.fc2 = nn.Linear(in_channels, 1)

    def transform_theta(self, in_theta):
        theta = torch.zeros(in_theta.size(0), 2, 3, device=self.device)
        theta[:,0,0] = torch.sigmoid(in_theta[:,0])
        theta[:,1,1] = torch.sigmoid(in_theta[:,1])
        theta[:,0,2] = torch.tanh(in_theta[:,2])
        theta[:,1,2] = torch.tanh(in_theta[:,3])
        return theta

    def forward(self, x):
        batch_size = x.size(0)
        stn_feature = x * self.channel_attr(x) + x
        theta = self.fc1(self.avgpool(stn_feature).view(batch_size, -1)).view(-1, 4)
        theta = self.transform_theta(theta)

        grid = F.affine_grid(theta, stn_feature.size())
        stn_feature = F.grid_sample(stn_feature, grid)
        stn_feature = self.avgpool(stn_feature)
        stn_feature = stn_feature.view(batch_size, -1)
        stn_feature = self.fc2(stn_feature)
        return stn_feature

class BaselineALM(nn.Module):
    __model_factory = {
        'resnet50': torchvision.models.resnet50,
        'resnet101': torchvision.models.resnet101
    }
    def __init__(self, num_classes, backbone='resnet50', pretrained=True, device=torch.device('cuda')):
        super(BaselineALM, self).__init__()
        self.num_classes = num_classes
        
        resnet = self.__model_factory[backbone](pretrained=pretrained)
        self.base = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

        self.classifier = nn.ModuleList()
        for _ in range(num_classes):
            self.classifier.append(ALM(2048, device))

    def forward(self, x):
        x = self.base(x)
        # x.size = (batch_size, 2048, 16, 8)
        out = []
        for i in range(self.num_classes):
            out.append(self.classifier[i](x))
        return torch.cat(out, dim=1)

if __name__ == "__main__":
    model = BaselineALM(26).to(torch.device('cuda'))
    batch = torch.rand((4, 3, 256, 128)).to(torch.device('cuda'))
    out = model(batch)
    pass