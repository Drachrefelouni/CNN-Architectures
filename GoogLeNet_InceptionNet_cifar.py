#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 15:41:38 2023

@author: achref
"""
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
 

class Inception(nn.Module):
    def __init__(self, in_channels,out_1_1,red_3_3,out_3_3,red_5_5,out_5_5,out_1_1_pool):
        super(Inception,self).__init__()
        
        self.branch1 = conv_block(in_channels, out_1_1, kernel_size = 1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3_3, kernel_size = 1),
            conv_block(red_3_3, out_3_3, kernel_size = 3, padding = 1)
            )
        
        
        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5_5, kernel_size = 1),
            conv_block(red_5_5, out_5_5, kernel_size = 5, padding = 2)
            )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1),
            conv_block(in_channels, out_1_1_pool, kernel_size = 1)
            )

    def forward(self,x):
        return torch.cat([self.branch1(x),self.branch2(x),self.branch3(x),self.branch4(x)],1)
        

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block,self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        
        
    def forward(self,x):
        return self.relu(self.batchnorm(self.conv(x)))


class GoogLeNet(nn.Module):
    def __init__(self, in_channels=3, num_classes = 1000):
        super(GoogLeNet,self).__init__()
        
        
        self.conv1 = conv_block(in_channels=in_channels, out_channels = 192, kernel_size = 3 ,stride= 1,padding = 1)
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride= 1, padding = 1)
        self.conv2 = conv_block(64 , 192 , kernel_size = 3, padding = 1)
        
        
        
        
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128,128, 192, 32, 96, 64)
        self.maxpool3 =  nn.MaxPool2d(kernel_size=3, stride= 2, padding = 1)


        self.inception4a = Inception( 480, 192,  96, 208, 16 , 48,  64)
        self.inception4b = Inception( 512, 160, 112, 224, 24 , 64,  64)
        self.inception4c = Inception( 512, 128, 128, 256, 24 , 64,  64)
        self.inception4d = Inception( 512, 112, 114, 288, 32,  64,  64)
        self.inception4e = Inception( 528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 =  nn.MaxPool2d(kernel_size=3, stride= 2, padding = 1)
        
        
        self.inception5a = Inception( 832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception( 832, 384, 192, 384, 48, 128, 128)
        
        
        self.avgpool4 =  nn.AvgPool2d(kernel_size=8, stride= 1)
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(1024, 1000)


    def forward(self,x):
        
        x = self.conv1(x)
    #    x = self.maxpool1(x)
     #   x = self.conv2(x)
        print(x.shape)

     #   x = self.maxpool2(x)
     #   print(x.shape)



        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        
        
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool4(x)
        x = x.reshape(x.shape[0],-1)
        x = self.dropout(x)
        x = self.fc1(x)
        
        return x



def save_checkpoint(state, filename="my_checkepoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state,filename)
    
def load_checkpoint(checkpoint):
    print("=> loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
 
    
    
device=torch.device('cude' if torch.cuda.is_available() else 'cpu')

in_channels=3
num_classes=10
lr=0.001
bach_size= 100
epoches= 5
load_model=False



train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
test_transform = transforms.Compose([transforms.ToTensor()])





train_dataset= datasets.CIFAR10(root='dataset/',train=True,transform=train_transform,download=True)
train_loader=DataLoader(dataset=train_dataset,batch_size=bach_size,shuffle=True)


test_dataset= datasets.CIFAR10(root='dataset/',train=False,transform=test_transform,download=True)
test_loader=DataLoader(dataset=test_dataset,batch_size=bach_size,shuffle=True)


model=GoogLeNet()

 

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
        print(data.shape)
        
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        