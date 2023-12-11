#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 17:50:14 2023

@author: achref
"""
import torch
import torch.nn as nn
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
import tqdm
from torchvision.utils import save_image

class VariationlAutoEncoder(nn.Module):
    
    
    def __init__(self, input_dim=784, h_dim=200, z_dim=20):
        super().__init__()
        
        #encoder
        self.img_2hid = nn.Linear(input_dim, h_dim)
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_2sigma = nn.Linear(h_dim, z_dim)
        #decoder
        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.hid_2img = nn.Linear(h_dim, input_dim)
        
        
        self.relu = nn.ReLU()
        
    def encode(self,x):
        h = self.relu(self.img_2hid(x))
        mu = self.hid_2mu(h) 
        sigma = self.hid_2sigma(h)

        return mu, sigma
        
    def decode(self,x):
        h= self.relu(self.z_2hid(x))
        return torch.sigmoid(self.hid_2img(h))
        
        
    def forward(self,x):
        mu, sigma = self.encode(x)
       
        epsilon = torch.randn_like(sigma)
        z_reparam = mu + sigma* epsilon
    
        x_reconstructed = self.decode(z_reparam)
        return x_reconstructed, mu, sigma
    
device=torch.device('cude' if torch.cuda.is_available() else 'cpu')

input_dim= 784
H_dim = 200
Z_dim = 20
lr= 0.001
bach_size= 32
epoches= 5

train_dataset= datasets.MNIST(root='dataset/',train=True,transform=transforms.ToTensor(),download=True)
train_loader=DataLoader(dataset=train_dataset,batch_size=bach_size,shuffle=True)


test_dataset= datasets.MNIST(root='dataset/',train=False,transform=transforms.ToTensor(),download=True)
test_loader=DataLoader(dataset=test_dataset,batch_size=bach_size,shuffle=True)


model = VariationlAutoEncoder().to(device)
loss_fn = nn.BCELoss(reduction = 'sum')
optimizer=optim.Adam(model.parameters(),lr)





for epoch in range(epoches):
    losses=[]

    for batch_idx, (data,targets) in enumerate(train_loader):

        # forward pass
        data = data.reshape(-1,28*28)
        x_reconst , mu, sigma = model(data)
        
        #loss compute
        reconst_loss = loss_fn(x_reconst,data)
        kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
        
        # backpropagation
        loss = reconst_loss + kl_div
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print(f'Loss of ech was {loss: .5f}')

    mean_loss= sum(losses)/len(losses)
    print(f'Loss of ech {epoch} was {mean_loss: .5f}')


def inference(digit, num_examples=1):
    """
    Generates (num_examples) of a particular digit.
    Specifically we extract an example of each digit,
    then after we have the mu, sigma representation for
    each digit we can sample from that.

    After we sample we can run the decoder part of the VAE
    and generate examples.
    """
    images = []
    idx = 0
    for x, y in train_dataset:
        if y == idx:
            images.append(x)
            idx += 1
        if idx == 10:
            break

    encodings_digit = []
    for d in range(10):
        with torch.no_grad():
            mu, sigma = model.encode(images[d].view(1, 784))
        encodings_digit.append((mu, sigma))

    mu, sigma = encodings_digit[digit]
    for example in range(num_examples):
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        out = model.decode(z)
        out = out.view(-1, 1, 28, 28)
        save_image(out, f"generated_{digit}_ex{example}.png")

for idx in range(10):
    inference(idx, num_examples=5)
    
























