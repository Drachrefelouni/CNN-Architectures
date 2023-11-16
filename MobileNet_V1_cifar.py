#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 17:49:35 2023

@author: achref
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
class MobileNetV1(nn.Module):
    def __init__(self, ch_in, n_classes):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            print(inp)
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
                )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                # pw
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
                )

        self.model = nn.Sequential(
            conv_bn(ch_in, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

    # model check

    
device=torch.device('cude' if torch.cuda.is_available() else 'cpu')

in_channels=3
num_classes=10
lr=0.001
bach_size= 10
epoches= 5
load_model=False




train_dataset= datasets.CIFAR10(root='dataset/',train=True,transform=transforms.ToTensor(),download=True)
train_loader=DataLoader(dataset=train_dataset,batch_size=bach_size,shuffle=True)


test_dataset= datasets.CIFAR10(root='dataset/',train=False,transform=transforms.ToTensor(),download=True)
test_loader=DataLoader(dataset=test_dataset,batch_size=bach_size,shuffle=True)


model= MobileNetV1(ch_in=3, n_classes=1000)
 

print(model)
model.to(device)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr)

 

for epoch in range(epoches):
    losses=[]
    
 
    
    for batch_idx, (data,targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        
        #forward
        scores = model(data)
        loss = criterion(scores,targets)
        losses.append(loss.item())
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        
        print(f'Loss of ech was {loss: .5f}')

        #gradient descent
        optimizer.step()
    mean_loss= sum(losses)/len(losses)
    print(f'Loss of ech {epoch} was {mean_loss: .5f}')
    
#check acc

def check_acc(loader,model):
    
    if loader.dataset.train :
        print('checking acc on training data')
    else:
        print('checking acc on test data')
    num_correct=0
    num_samples=0
    model.eval()
    with torch.no_grad():
        for x,y in loader :
            x=x.to(device=device)
            y=y.to(device=device)
            
            scores = model(x)
            _, preds = scores.max(1)
            num_correct+=(preds==y).sum()
            num_samples+=preds.size(0)
            
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100}')
    
    model.train()
 

check_acc(train_loader, model)
check_acc(test_loader, model)
    
    