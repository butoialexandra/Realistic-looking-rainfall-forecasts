import argparse
import os
import numpy as np
import math
import warnings

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from datetime import datetime

from dataset import Dataset
from generator import Generator, GeneratorA



parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr_gen", type=float, default=0.0005, help="adam: learning rate")
parser.add_argument("--lr_disc", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=1000, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=10, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

# img_shape = (opt.channels, opt.img_size, opt.img_size)
input_shape = (127, 188)
output_shape = (295, 427)
# TODO: make sure we didn't mix up x, y from dataset
in_pixels = int(np.prod(input_shape))
out_pixels = int(np.prod(output_shape))

if torch.cuda.is_available():
    cuda = True
    print("Using CUDA")
else:
    cuda = False
    print("NOT using CUDA")

device = torch.device("cuda:0" if cuda else "cpu")


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        in_pixels = int(np.prod(input_shape))
        #out_pixels = int(np.prod(output_shape))

        self.model = nn.Sequential(
            # nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5, stride=1),
            # nn.Linear(out_pixels + in_pixels, 512),
            nn.Linear(in_pixels, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, observation):
        # Concatenate label embedding and image to produce input
        # observation = observation.view(observation.size(0), -1)
        # print(observation.shape, img.shape)
        # d_in = torch.cat((img.view(img.size(0), -1), observation), -1)
        # d_in = torch.stack((observation, img), dim=1)
        d_in = img.view(img.size(0), -1)
        validity = self.model(d_in)
        validity = torch.sigmoid(validity)
        return validity


# Loss functions
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = GeneratorA()
discriminator = Discriminator()
datetimestr = datetime.now().strftime("%d-%b-%Y-%H:%M")
os.makedirs(f"images_{type(generator).__name__}_{datetimestr}", exist_ok=True)
if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

training_params = {"batch_size": opt.batch_size, "shuffle": True, "num_workers": 0}
training_data = Dataset(device=device)
train_idx, test_idx = training_data.train_test_split_ids(how='seq')
training_data.select_indices(train_idx, shuffle=False)  # TODO: this might cause problems!
training_generator = torch.utils.data.DataLoader(training_data, **training_params)


validation_params = {"batch_size": opt.batch_size, "shuffle": False, "num_workers": 0}
validation_data = Dataset(device=device)
validation_data.select_indices(test_idx, shuffle=False)  # here order doesn't matter
validation_generator = torch.utils.data.DataLoader(validation_data, **validation_params)


print("Generators OK")

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_gen, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr_disc, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_image_(training_data, n_row, batches_done):
    """Saves a grid of generated images"""
    n_row = 1 # FIXME
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    y_pred, y_real = training_data.get_x_y_by_id(training_data.selected_indices[0])  # TODO: fix first date from 201805
    y_pred = torch.tensor(y_pred, device=device).repeat(n_row, 1, 1)
    y_real = torch.tensor(y_real, device=device).repeat(n_row, 1, 1)
    # print(y_real.size())
    # print(z.size())
    # labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, y_pred)
    gen_imgs = gen_imgs.unsqueeze(1)
    # print(gen_imgs.data.size())
    # data = torch.cat((gen_imgs.data.squeeze(0), y_real), dim=0)
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=2, normalize=True)


def sample_image(training_data, n_row, batches_done):
    n_row = 1 # FIXME
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    y_pred, y_real = training_data.get_x_y_by_id(training_data.selected_indices[0])  # TODO: fix first date from 201805
    y_pred = torch.tensor(y_pred, device=device).repeat(n_row, 1, 1) / 10
    y_real = torch.tensor(y_real, device=device).repeat(n_row, 1, 1) / 10
    gen_imgs = generator(z, y_pred)

    fig = plt.figure(figsize=(6, 3.2))
    ax = fig.add_subplot(121)
    ax.set_title('Output')
    plt.imshow(gen_imgs.squeeze().detach().cpu())

    ax = fig.add_subplot(122)
    ax.set_title('Observation')
    plt.imshow(y_real.squeeze().detach().cpu())
    plt.colorbar(orientation='horizontal')
    plt.savefig("images/%d.png" % batches_done)


# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    print("Starting epoch %d" % epoch)

    for i, (pred_imgs, real_imgs) in enumerate(training_generator):

        batch_size = pred_imgs.shape[0]
        if torch.any(pred_imgs.isnan()):
            warnings.warn("Skipping batch with nan value")
            continue

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(real_imgs.type(FloatTensor)) / 10
        pred_imgs = Variable(pred_imgs.type(FloatTensor)) / 10

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        # gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

        # Generate a batch of images
        gen_imgs = generator(z, pred_imgs)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, pred_imgs)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_imgs, pred_imgs)
        print("Validity real:", validity_real)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        fake_imgs = gen_imgs.detach()
        # fake_imgs = torch.zeros_like(fake_imgs) # TODO: remove me!!!!!
        validity_fake = discriminator(fake_imgs, pred_imgs)
        print("Validity fake:", validity_fake)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(training_generator), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(training_generator) + i
        if batches_done % opt.sample_interval == 0:
            sample_image(training_data, n_row=10, batches_done=batches_done)
    # sample_image(training_data, n_row=10, batches_done=epoch)
