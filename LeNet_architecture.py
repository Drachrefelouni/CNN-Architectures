#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 12:40:30 2023

@author: achref
"""
import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self,in_channels=3,num_classes=10):
        super(LeNet,self).__init__()
        
        
 
        self.relu  = nn.ReLU()
        self.conv1  = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5,stride=1,padding=0)
        self.conv2  = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5,stride=1,padding=0)
        self.conv3  = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5,stride=1,padding=0)
        
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, 10)
        
        self.pool = nn.AvgPool2d(kernel_size=(2,2), stride = (2,2))
        
        
    def forward(self,x):

        x = self.pool(self.relu(self.conv1(x))) 
        x = self.pool(self.relu(self.conv2(x))) 
        x = self.relu(self.conv3(x))

        x = x.reshape(x.shape[0],-1)
        
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        
        return x
        
    
x = torch.randn(64,3,32,32)
model = LeNet()
print(model(x).shape)