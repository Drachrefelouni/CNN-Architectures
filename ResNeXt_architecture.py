#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 14:36:50 2023

@author: achref
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 15:57:06 2023

@author: achref
"""
import torch 
import torch.nn as nn

# ConvBlock
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, group = 1, biais = 0):
        super().__init__()
        self.c = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,groups=group,bias=biais)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        return self.bn(self.c(x))

# Bottleneck ResidualBlock 
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, first=False, cordinality = 32):
        super().__init__()
        C = cordinality
        res_channels = out_channels // 2 
        
        
        self.downsample = stride == 2 or first
        self.c1 = ConvBlock(in_channels, res_channels, 1, 1, 0)

        self.c2 = ConvBlock(res_channels, res_channels, 3, stride, 1, C)

        self.c3 = ConvBlock(res_channels, out_channels, 1, 1, 0)

        if self.downsample:
            self.p = ConvBlock(in_channels, out_channels, 1, stride, 0)
            
        self.relu = nn.ReLU()        

    def forward(self,x):
        f = self.relu(self.c1(x))
        f = self.relu(self.c2(f))
        f = self.c3(f)
        
        if self.downsample:
            x=self.p(x)
        
        
        h = self.relu(torch.add(f,x))
        return h
        
        
        
# ResNetx
class ResNet(nn.Module):
    def __init__(
        self, 
        config_name : int, 
        in_channels=3, 
        classes=1000
        ):
        super().__init__()

        configurations = {
            50 : [3, 4, 6, 3],
            101 : [3, 4, 23, 3],
            152 : [3, 8, 36, 3]
        }

        no_blocks = configurations[config_name]

        out_features = [256, 512, 1024, 2048]
        self.blocks = nn.ModuleList([ResidualBlock(64, 256, 1, True)])

        for i in range(len(out_features)):
            if i > 0:

                self.blocks.append(ResidualBlock(out_features[i-1], out_features[i], 2))
            for _ in range(no_blocks[i]-1):

                self.blocks.append(ResidualBlock(out_features[i], out_features[i],1))

        self.conv1 = ConvBlock(in_channels, 64, 7, 2, 3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, classes)

        self.relu = nn.ReLU()

        self.init_weight()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        for block in self.blocks:
            x = block(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def init_weight(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)


if __name__ == "__main__":
    config_name = 50 # 50-layer
    resnet50 = ResNet(50)
    image = torch.rand(1, 3, 224, 224)
