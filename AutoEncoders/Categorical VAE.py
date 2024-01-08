#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 18:03:16 2024

@author: achref
"""

from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.distributions import kl_divergence
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
                   ])),
        batch_size=128, shuffle=True)


test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='../data', train=False, download=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
                       ])),
    batch_size=1, shuffle=True)

def sample_gumbel(shape, eps=1e-20):
    unif = torch.rand(*shape).to(device)
    g = -torch.log(-torch.log(unif + eps))
    return g

def sample_gumbel_softmax(logits, temperature):
    """
        Input:
        logits: Tensor of log probs, shape = BS x k
        temperature = scalar
        
        Output: Tensor of values sampled from Gumbel softmax.
                These will tend towards a one-hot representation in the limit of temp -> 0
                shape = BS x k
    """
    g = sample_gumbel(logits.shape)
    h = (g + logits)/temperature
    h_max = h.max(dim=-1, keepdim=True)[0]
    h = h - h_max
    cache = torch.exp(h)
    y = cache / cache.sum(dim=-1, keepdim=True)
    return y

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = 1.
        self.K = 10
        self.N = 30
        self.create_encoder()
        self.create_decoder()
        
    def create_encoder(self):
        """
            Input for the encoder is a BS x 784 tensor
            Output from the encoder are the log probabilities of the categorical distribution
        """
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.K*self.N)
        )
        
    def create_decoder(self):
        """
            Input for the decoder is a BS x N*K tensor
            Output from the decoder are the log probabilities of the bernoulli pixels
        """
        self.decoder = nn.Sequential(
            nn.Linear(self.N*self.K, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.LogSigmoid()
        )
        
    def sample(self, img):
        with torch.no_grad():
            logits_nz = self.encoder(img)
            logits_z = F.log_softmax(logits_nz.view(-1, self.N, self.K), dim=-1)
            latent_vars = sample_gumbel_softmax(logits_z, self.temperature)
            logits_x = self.decoder(latent_vars)
            dist_x = torch.distributions.Bernoulli(logits=logits_x)
            sampled_img = dist_x.sample((1,))
            
        return sampled_img.cpu().numpy()
        
    def forward(self, img, anneal=1.):
        """
            Input: 
            img: Tensor of shape BS x 784
        """
        # Encoding
        logits_nz = self.encoder(img)
        logits_z = F.log_softmax(logits_nz.view(-1, self.N, self.K), dim=-1)
        posterior_dist = torch.distributions.Categorical(logits=logits_z)
        prior_dist = torch.distributions.Categorical(probs=torch.ones_like(logits_z)/self.K)
        
        # Sampling
        latent_vars = sample_gumbel_softmax(logits_z, self.temperature).view(-1, self.N*self.K)
        
        # Decoding
        logits_x = self.decoder(latent_vars)
        dist_x = torch.distributions.Bernoulli(logits=logits_x)

        # Losses
        ll = dist_x.log_prob(img).sum(dim=-1)
#         kl1 = posterior_dist.probs * (logits_z - torch.log(torch.ones_like(logits_z)/self.K))
        kl = kl_divergence(posterior_dist, prior_dist).sum(-1)
        assert torch.all(kl > 0)
        assert torch.all(ll < 0)
        elbo = ll - kl
        loss = -elbo.mean()
        return loss
def sample(model, img):
    with torch.no_grad():
        logits_nz = model.encoder(img)
        logits_z = F.log_softmax(logits_nz.view(-1, model.N, model.K), dim=-1)
        latent_vars = sample_gumbel_softmax(logits_z, model.temperature).view(-1, model.N*model.K)
        logits_x = model.decoder(latent_vars)
        dist_x = torch.distributions.Bernoulli(logits=logits_x)
        sampled_img = dist_x.sample((1,))

    return sampled_img.cpu().numpy()


def plot_img(model):
    for batch_idx, (data, target) in enumerate(test_loader):
        img_flat = sample(model, data.view(-1, 28*28).to(device))
        plt.figure()
        plt.subplot(121)
        plt.imshow(img_flat.reshape(28,28))
        plt.subplot(122)
        plt.imshow(data.reshape(28,28))
        plt.show()
        break
    
    
    
def train(model, optimizer, maxiters):
    iters = 0
    while iters < maxiters:
        for batch_idx, (data, target) in enumerate(train_loader):
            iters+=1
#             anneal = min(1., epoch*.1)
            optimizer.zero_grad()
            data = data.to(device)
            loss = model(data.view(-1, 28*28))
    #             neg_elbo = -elbo
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                model.temperature = np.maximum(model.temperature * np.exp(-ANNEAL_RATE * batch_idx), temp_min)
                print("New Model Temperature: {}".format(model.temperature))
            if iters % 100 == 0:
                plot_img(model)
                print('Train Iteration: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    iters, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))   

device=torch.device('cuda')

model = VAE().to(device)
optimizer = Adam(model.parameters(), lr=1e-3)
temp_min = 0.5
ANNEAL_RATE = 0.00003
train(model, optimizer, maxiters=10000)