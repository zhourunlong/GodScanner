import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import numpy as np
from PIL import ImageFont, ImageDraw, ImageOps
from random import randint
import matplotlib.font_manager as fm
import pandas as pd
from IPython import display

import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from tqdm import tqdm

from nets import GeneratorNet, DiscriminatorNet
from dataset import build_dataset

def calculate_psnr(img1, img2, max_value=255):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))

train_set, valid_set = build_dataset()

bs = 25
train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=bs, shuffle=True)

discriminator = DiscriminatorNet(in_channels=2)
generator = GeneratorNet(in_channels=1, out_channels=1)

discriminator.cuda()
generator.cuda()

# Optimizers
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss function
loss1 = nn.MSELoss()
loss2 = nn.BCELoss()
loss3 = nn.BCEWithLogitsLoss()

num_epochs = 500
loss_lambda = 500

for epoch in range(num_epochs):

    generator.train()

    with tqdm(train_loader, desc="training") as pbar:
        
        running_loss_d, running_loss_g = [], []

        for (degraded_batch, clean_batch) in pbar:

            real_data = clean_batch.float().cuda()
            noised_data = degraded_batch.float().cuda()
            
            # 1. Train Discriminator
            # Generate fake data
            fake_data = generator(noised_data)

            # Reset gradients
            d_optimizer.zero_grad()
            
            # 1.1 Train on Real Data
            prediction_real = discriminator(real_data, noised_data)

            # Calculate error and backpropagate
            real_data_target = torch.ones_like(prediction_real)
            loss_real = loss1(prediction_real, real_data_target)

            # 1.2 Train on Fake Data, you would need to add one more component
            prediction_fake = discriminator(fake_data, noised_data)

            # Calculate error and backpropagate
            fake_data_target = torch.zeros_like(prediction_real)
            loss_fake = loss1(prediction_fake, fake_data_target)

            loss_d = (loss_real + loss_fake)/2
            loss_d.backward(retain_graph=True)
            
            # 1.3 Update weights with gradients
            d_optimizer.step()
    
            # 2. Train Generator
            g_optimizer.zero_grad()

            # Sample noise and generate fake data
            prediction = discriminator(fake_data, real_data)
            
            # Calculate error and backpropagate
            real_data_target = torch.ones_like(prediction)
            #import pdb; pdb.set_trace();

            loss_g1 = loss1(prediction, real_data_target)
            loss_g2 = loss1(fake_data, real_data) * loss_lambda
            loss_g = loss_g1 + loss_g2

            loss_g.backward()

            # Update weights with gradients
            g_optimizer.step()
                    
            # Log error
            running_loss_d.append(loss_d.item())
            running_loss_g.append(loss_g.item())

            pbar.set_description("Epoch: %d, Loss_d: %0.5f, Loss_g: %0.5f," % (epoch, np.mean(running_loss_d), np.mean(running_loss_g)))
    
    state_dict = dict(d=discriminator.state_dict(), g=generator.state_dict(), d_optimizer=d_optimizer.state_dict(), g_optimizer=g_optimizer.state_dict())
    torch.save(state_dict, "model/state_dict_epoch%d.pt" % (epoch))
    
    generator.eval()

    degraded, predicted, clean = [], [], []

    with tqdm(valid_loader, desc="evaluating") as pbar:
        for (degraded_batch, clean_batch) in pbar:

            with torch.no_grad():
                output = generator(degraded_batch.float().cuda())

            if (len(degraded) < 2):
                degraded.append(degraded_batch.view(-1, 256))
                predicted.append(output.detach().view(-1, 256).cpu())
                clean.append(clean_batch.view(-1, 256))

    _, ax = plt.subplots(1, 3, figsize=(30, 10 * bs))
    ax[0].imshow(np.concatenate(degraded), cmap='gray')
    ax[0].title.set_text('Degrade Image')
    ax[1].imshow(np.concatenate(predicted), cmap='gray')
    ax[1].title.set_text('Predicted Image')
    ax[2].imshow(np.concatenate(clean), cmap='gray')
    ax[2].title.set_text('Clean Image')
    plt.savefig("result/evaluation%d.png" % (epoch))
