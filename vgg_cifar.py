#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 17:14:27 2023

@author: achref
"""
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
import torchvision
VGG16 = [64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M']

class VGG_net(nn.Module):
    def __init__(self,in_channels=3,num_classes=10):
        super(VGG_net,self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG16)
        
        self.fcs= nn.Sequential(
                    nn.Linear(512,100),
                    nn.ReLU(),
                    nn.Linear(100, 10)
            )
        
    def forward(self,x):
        x =self.conv_layers(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fcs(x) 
        return x
        
        
    
    def create_conv_layers(self,architectures):
        layers= []
        in_channels=self.in_channels
        
        for x in architectures:
            if type(x)== int:
                out_channels = x
                print(in_channels)
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]
                
                in_channels = x
            elif x== 'M':
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]
                
                
        return nn.Sequential(*layers)


def save_checkpoint(state, filename="my_checkepoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state,filename)
    
def load_checkpoint(checkpoint):
    print("=> loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
class Identity(nn.Module):
    def __init__(self):
        super(Identity,self).__init__()
    
    def forward(self,x):
        return x
    
    
device=torch.device('cude' if torch.cuda.is_available() else 'cpu')

in_channels=3
num_classes=10
lr=0.001
bach_size= 1024
epoches= 5
load_model=False




train_dataset= datasets.CIFAR10(root='dataset/',train=True,transform=transforms.ToTensor(),download=True)
train_loader=DataLoader(dataset=train_dataset,batch_size=bach_size,shuffle=True)


test_dataset= datasets.CIFAR10(root='dataset/',train=False,transform=transforms.ToTensor(),download=True)
test_loader=DataLoader(dataset=test_dataset,batch_size=bach_size,shuffle=True)


model=VGG_net()

 

print(model)
model.to(device)
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
        print(targets.shape[0])

        
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        