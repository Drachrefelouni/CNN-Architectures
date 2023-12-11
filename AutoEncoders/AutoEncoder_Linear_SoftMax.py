#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 13:25:39 2023

@author: achref
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms



class Autoencoder(nn.Module):
    
     def __init__(self):
         super().__init__()
         self.encoder = nn.Sequential(
             nn.Linear(28*28,128),
             nn.ReLU(),
             nn.Linear(128,64),
             nn.ReLU(),
             nn.Linear(64, 12),
             nn.ReLU(),
             nn.Linear(12, 3)
             )
         
         self.dencoder = nn.Sequential(
             nn.Linear(3,12),
             nn.ReLU(),
             nn.Linear(12,64),
             nn.ReLU(),
             nn.Linear(64, 128),
             nn.ReLU(),
             nn.Linear(128,10),
     

             )
     def forward(self,x):
         return self.dencoder(self.encoder(x))

def save_checkpoint(state, filename="my_checkepoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state,filename)
    
def load_checkpoint(checkpoint):
    print("=> loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    
    
    
device=torch.device('cude' if torch.cuda.is_available() else 'cpu')

num_classes=10
lr= 3e-4
bach_size= 64
epoches= 5
load_model=False


train_dataset= datasets.MNIST(root='dataset/',train=True,transform=transforms.ToTensor(),download=True)
train_loader=DataLoader(dataset=train_dataset,batch_size=bach_size,shuffle=True)


test_dataset= datasets.MNIST(root='dataset/',train=False,transform=transforms.ToTensor(),download=True)
test_loader=DataLoader(dataset=test_dataset,batch_size=bach_size,shuffle=True)


model=Autoencoder().to(device)

criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr)


if load_model:
    load_checkpoint(torch.load("my_checkepoint.pth.tar"))

for epoch in range(epoches):
    losses=[]
    
    if epoch%3 ==0:
        checkpoint = {'state_dict' : model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)
    
    for batch_idx, (data,targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)
                
        data = data.reshape(-1,28*28)
        #forward
        scores = model(data)
        loss = criterion(scores,targets)
        losses.append(loss.item())
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        
        
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
            x = x.reshape(-1,28*28)

            scores = model(x)
            _, preds = scores.max(1)
            num_correct+=(preds==y).sum()
            num_samples+=preds.size(0)
            
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100}')
    
    model.train()
 

check_acc(train_loader, model)
check_acc(test_loader, model)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        