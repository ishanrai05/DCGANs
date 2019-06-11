import argparse

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

from parameters import (num_z, num_feat_gen, num_feat_disc, \
                        num_channels, image_size, batch_size, lr, beta1)
from utils import weights_init
from train import train
from generator import Generator
from discriminator import Discriminator


device = torch.device("cpu")


# '''
parser = argparse.ArgumentParser(description='PyTorch Fer2013 CNN Training')
parser.add_argument('--use_cuda', type=bool, default=False, help='device to train on')
parser.add_argument('--samples', type=bool, default=False, help='See sample images')
parser.add_argument('--view_data_counts', type=bool, default=False, help='Visualize data distribution')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train on')
parser.add_argument('--train', default=True, type=bool, help='train the model')

opt = parser.parse_args()


if opt.use_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# '''


root = 'celeba/'

# Create the dataset
dataset = datasets.ImageFolder(root=root,
                              transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=4)


if opt.samples:
    # Plot some training images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(torchvision.utils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

if opt.train:

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(disc.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(gen.parameters(), lr=lr, betas=(beta1, 0.999))

    gen = Generator(num_z, num_feat_gen, num_channels).to(device)
    disc = Discriminator(num_channels, num_feat_disc).to(device)

    gen.apply(weights_init)
    disc.apply(weights_init)

    print (gen)
    print (disc)

    criterion = nn.BCELoss()

    G_losses, D_losses, img_list = train(5, num_z, dataloader, \
        gen, disc, device, criterion, optimizerG, optimizerD)

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(torchvision.utils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.show()