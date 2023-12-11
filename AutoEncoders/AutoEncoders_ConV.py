#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 17:25:05 2023

@author: achref
"""
import torch
from torch import nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5),(0.5))
    ])
transform = transforms.ToTensor()

mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset=mnist_data, batch_size=64, shuffle= True)


class CNNBLOCK(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride, padding):
        super(CNNBLOCK, self).__init__()
        self.cnn= nn.Conv2d(in_channels, out_channels, kernel_size,stride,padding)
        self.relu = nn.ReLU()
        
    
    def forward(self, x):
        return self.relu(self.cnn(x))
    
    
  
    
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            CNNBLOCK(1, 16, 3,2,1),
            CNNBLOCK(16, 32, 3,2,1),
            nn.Conv2d(32, 64, 7)
            )
        
        self.dencoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3,2,1,1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3,2,1,1),
            nn.Sigmoid()
            )
    def forward(self,x):
        return self.dencoder(self.encoder(x))
    
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-5)

num_epochs = 10
outputs = []
for epoch in range(num_epochs):
    for(img,_) in data_loader:
       # img = img.reshape(-1,28*28)
        recon = model(img)
        loss = criterion(recon,img)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
    outputs.append((epoch,img,recon))


for k in range(0, num_epochs, 4):
    plt.figure(figsize=(9, 2))
    plt.gray()
    imgs = outputs[k][1].detach().numpy()
    recon = outputs[k][2].detach().numpy()
    for i, item in enumerate(imgs):
        if i >= 9: break
        plt.subplot(2, 9, i+1)
        # item = item.reshape(-1, 28,28) # -> use for Autoencoder_Linear
        # item: 1, 28, 28
        plt.imshow(item[0])
            
    for i, item in enumerate(recon):
        if i >= 9: break
        plt.subplot(2, 9, 9+i+1) # row_length + i + 1
        # item = item.reshape(-1, 28,28) # -> use for Autoencoder_Linear
        # item: 1, 28, 28
        plt.imshow(item[0])