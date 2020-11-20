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
from torch.nn import init
import matplotlib.pyplot as plt

from dataset import Dataset


os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
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

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, dilation=1, groups=1, bias=False,
                 do_norm=True, norm = 'batch', do_activation = True): # bias default is True in Conv2d
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.leakyRelu = nn.LeakyReLU(0.2, True)
        self.do_norm = do_norm
        self.do_activation = do_activation
        if do_norm:
            if norm == 'batch':
                self.norm = nn.BatchNorm2d(out_channels)
            elif norm == 'instance':
                self.norm = nn.InstanceNorm2d(out_channels)
            elif norm == 'none':
                self.do_norm = False
            else:
                raise NotImplementedError("norm error")

    def forward(self, x):
        if self.do_activation:
            x = self.leakyRelu(x)

        x = self.conv(x)

        if self.do_norm:
            x = self.norm(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False,
                 do_norm=True, norm = 'batch',do_activation = True, dropout_prob=0.0):
        super(DecoderBlock, self).__init__()

        self.convT = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.relu = nn.ReLU()
        self.dropout_prob = dropout_prob
        self.drop = nn.Dropout2d(dropout_prob)
        self.do_norm = do_norm
        self.do_activation = do_activation
        if do_norm:
            if norm == 'batch':
                self.norm = nn.BatchNorm2d(out_channels)
            elif norm == 'instance':
                self.norm = nn.InstanceNorm2d(out_channels)
            elif norm == 'none':
                self.do_norm = False
            else:
                raise NotImplementedError("norm error")

    def forward(self, x):
        if self.do_activation:
            x = self.relu(x)

        x = self.convT(x)

        if self.do_norm:
           x = self.norm(x)

        if self.dropout_prob != 0:
            x= self.drop(x)

        return x

class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, bias = False, dropout_prob=0.5, norm = 'batch'):
        super(Generator, self).__init__()

        # 8-step encoder
        self.encoder1 = EncoderBlock(in_channels, 64, bias=bias, do_norm=False, do_activation=False)
        self.encoder2 = EncoderBlock(64, 128, bias=bias, norm=norm)
        self.encoder3 = EncoderBlock(128, 256, bias=bias, norm=norm)
        self.encoder4 = EncoderBlock(256, 512, bias=bias, norm=norm)
        self.encoder5 = EncoderBlock(512, 512, bias=bias, norm=norm)
        self.encoder6 = EncoderBlock(512, 512, bias=bias, norm=norm)
        self.encoder7 = EncoderBlock(512, 512, bias=bias, norm=norm)
        self.encoder8 = EncoderBlock(512, 512, bias=bias, do_norm=False)

        # 8-step UNet decoder
        self.decoder1 = DecoderBlock(512, 512, bias=bias, norm=norm)
        self.decoder2 = DecoderBlock(1024, 512, bias=bias, norm=norm, dropout_prob=dropout_prob)
        self.decoder3 = DecoderBlock(1024, 512, bias=bias, norm=norm, dropout_prob=dropout_prob)
        self.decoder4 = DecoderBlock(1024, 512, bias=bias, norm=norm, dropout_prob=dropout_prob)
        self.decoder5 = DecoderBlock(1024, 256, bias=bias, norm=norm)
        self.decoder6 = DecoderBlock(512, 128, bias=bias, norm=norm)
        self.decoder7 = DecoderBlock(256, 64, bias=bias, norm=norm)
        self.decoder8 = DecoderBlock(128, out_channels, bias=bias, do_norm=False)

    def forward(self, x):
        # 8-step encoder
        encode1 = self.encoder1(x)
        encode2 = self.encoder2(encode1)
        encode3 = self.encoder3(encode2)
        encode4 = self.encoder4(encode3)
        encode5 = self.encoder5(encode4)
        encode6 = self.encoder6(encode5)
        encode7 = self.encoder7(encode6)
        encode8 = self.encoder8(encode7)

        # 8-step UNet decoder
        decode1 = torch.cat([self.decoder1(encode8), encode7],1)
        decode2 = torch.cat([self.decoder2(decode1), encode6],1)
        decode3 = torch.cat([self.decoder3(decode2), encode5],1)
        decode4 = torch.cat([self.decoder4(decode3), encode4],1)
        decode5 = torch.cat([self.decoder5(decode4), encode3],1)
        decode6 = torch.cat([self.decoder6(decode5), encode2],1)
        decode7 = torch.cat([self.decoder7(decode6), encode1],1)
        decode8 = self.decoder8(decode7)
        final = nn.Tanh()(decode8)
        return final

class Discriminator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, bias = False, norm = 'batch', sigmoid=True):
        super(Discriminator, self).__init__()
        self.sigmoid = sigmoid

        # 70x70 discriminator
        self.disc1 = EncoderBlock(in_channels*2, 64, bias=bias, do_norm=False, do_activation=False)
        self.disc2 = EncoderBlock(64, 128, bias=bias, norm=norm)
        self.disc3 = EncoderBlock(128, 256, bias=bias, norm=norm)
        self.disc4 = EncoderBlock(256, 512, bias=bias, norm=norm, stride=1)
        self.disc5 = EncoderBlock(512, out_channels, bias=bias, stride=1, do_norm=False)

    def forward(self, x, ref):
        d1 = self.disc1(torch.cat([x, ref],1))
        d2 = self.disc2(d1)
        d3 = self.disc3(d2)
        d4 = self.disc4(d3)
        d5 = self.disc5(d4)
        if self.sigmoid:
            final = nn.Sigmoid()(d5)
        else:
            final = d5
        return final

# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#
#         in_pixels = int(np.prod(input_shape))
#         out_pixels = int(np.prod(output_shape))
#
#         self.model = nn.Sequential(
#             # nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5, stride=1),
#             # nn.Linear(out_pixels + in_pixels, 512),
#             nn.Linear(out_pixels, 512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(512, 512),
#             nn.Dropout(0.4),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(512, 512),
#             nn.Dropout(0.4),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(512, 1),
#         )
#
#     def forward(self, img, observation):
#         # Concatenate label embedding and image to produce input
#         # observation = observation.view(observation.size(0), -1)
#         # print(observation.shape, img.shape)
#         # d_in = torch.cat((img.view(img.size(0), -1), observation), -1)
#         # d_in = torch.stack((observation, img), dim=1)
#         d_in = img.view(img.size(0), -1)
#         validity = self.model(d_in)
#         validity = torch.sigmoid(validity)
#         return validity

# class Generator(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         def block(in_feat, out_feat, normalize=True):
#             layers = [nn.Linear(in_feat, out_feat)]
#             # if normalize:
#                 # layers.append(nn.BatchNorm1d(out_feat, 0.8))
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             return layers
#
#         self.in_pixels = int(np.prod(input_shape))
#         self.out_pixels = int(np.prod(output_shape))
#
#         self.model = nn.Sequential(
#             *block(1000 + 0
#             # self.in_pixels
#             , 1024, normalize=False),
#             *block(1024, 1024),
#             # *block(2560, 512),
#             # *block(512, 1024),
#             nn.Linear(1024, self.out_pixels),
#             # nn.Tanh()
#             nn.Sigmoid(),
#         )
#
#     def forward(self, noise, observation):
#         # Concatenate label embedding and image to produce input
#         # Keep only batch size and flatten everything else
#         observation = observation.view(observation.size(0), self.in_pixels)#-1)
#         # print("A",observation.size())
#         # gen_input = torch.cat((observation, noise), -1)
#         gen_input = noise
#         # print("B",gen_input.size())
#         img = self.model(gen_input)
#         # print("C",img.size())
#         img = img.view(img.size(0), *output_shape)
#         # print("D",img.size())
#         return img

def init_weights(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

# Loss functions
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator(bias=False, norm="batch", dropout_prob=0.5)
discriminator = Discriminator(bias=False, norm="batch", sigmoid=True)

# Initialize xavier
generator.apply(init_weights)
discriminator.apply(init_weights)

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
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_image_(training_data, n_row, batches_done):
    """Saves a grid of generated images"""
    n_row = 1 # FIXME
    # Sample noise
    # z = Variable(FloatTensor(np.random.normal(0, 1, (n_row, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    y_pred, y_real = training_data.get_x_y_by_id(training_data.selected_indices[0])  # TODO: fix first date from 201805
    y_pred = torch.tensor(y_pred, device=device).repeat(n_row, 1, 1)
    y_real = torch.tensor(y_real, device=device).repeat(n_row, 1, 1)
    # print(y_real.size())
    # print(z.size())
    # labels = Variable(LongTensor(labels))
    gen_imgs = generator(y_pred)
    gen_imgs = gen_imgs.unsqueeze(1)
    # print(gen_imgs.data.size())
    # data = torch.cat((gen_imgs.data.squeeze(0), y_real), dim=0)
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=2, normalize=True)


def sample_image(training_data, n_row, batches_done):
    n_row = 1 # FIXME
    # Sample noise
    # z = Variable(FloatTensor(np.random.normal(0, 1, (n_row, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    y_pred, y_real = training_data.get_x_y_by_id(training_data.selected_indices[0])  # TODO: fix first date from 201805
    y_pred = torch.tensor(y_pred, device=device).repeat(n_row, 1, 1) / 10
    y_real = torch.tensor(y_real, device=device).repeat(n_row, 1, 1) / 10
    gen_imgs = generator(y_pred)

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
        pred_imgs = pred_imgs.unsqueeze(1)
        real_imgs = real_imgs.unsqueeze(1)
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
        # z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        # gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

        # Generate a batch of images
        gen_imgs = generator(pred_imgs)

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
