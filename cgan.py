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


def norm_relu_layer(out_channel, do_norm, norm, relu):
    if do_norm:
        if norm == 'instance':
            norm_layer = nn.InstanceNorm2d(out_channel)
        elif norm == 'batch':
            norm_layer = nn.BatchNorm2d(out_channel)
        else:
            raise NotImplementedError("norm error")
    else:
        norm_layer = nn.Dropout2d(0)  # Identity

    if relu is None:
        relu_layer = nn.ReLU()
    else:
        relu_layer = nn.LeakyReLU(relu, inplace=True)

    return norm_layer, relu_layer

def Conv_Norm_ReLU(in_channel, out_channel, kernel, padding=0, dilation=1, groups=1, stride=1, bias=True,
                   do_norm=True, norm='batch', relu=None):
    """
    Convolutional -- Norm -- ReLU Unit
    :param norm: 'batchnorm' --> use BatchNorm2D, 'instancenorm' --> use InstanceNorm2D, 'none' --> Identity()
    :param relu: None -> Use vanilla ReLU; float --> Use LeakyReLU(relu)
    :input (N x in_channel x H x W)
    :return size same as nn.Conv2D
    """
    norm_layer, relu_layer = norm_relu_layer(out_channel, do_norm, norm, relu)

    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel, padding=padding, stride=stride,
                  dilation=dilation, groups=groups, bias=bias),
        norm_layer,
        relu_layer
    )

def Deconv_Norm_ReLU(in_channel, out_channel, kernel, padding=0, output_padding=0, stride=1, groups=1,
                     bias=True, dilation=1, do_norm=True, norm='batch'):
    """
    Deconvolutional -- Norm -- ReLU Unit
    :param norm: 'batchnorm' --> use BatchNorm2D, 'instancenorm' --> use InstanceNorm2D, 'none' --> Identity()
    :param relu: None -> Use vanilla ReLU; float --> Use LeakyReLU(relu)
    :input (N x in_channel x H x W)
    :return size same as nn.ConvTranspose2D
    """
    norm_layer, relu_layer = norm_relu_layer(out_channel, do_norm, norm, relu=None)
    return nn.Sequential(
        nn.ConvTranspose2d(in_channel, out_channel, kernel, padding=padding, output_padding=output_padding,
                           stride=stride, groups=groups, bias=bias, dilation=dilation),
        norm_layer,
        relu_layer
    )

class ResidualLayer(nn.Module):
    """
    Residual block used in Johnson's network model:
    Our residual blocks each contain two 3×3 convolutional layers with the same number of filters on both
    layer. We use the residual block design of Gross and Wilber [2] (shown in Figure 1), which differs from
    that of He et al [3] in that the ReLU nonlinearity following the addition is removed; this modified design
    was found in [2] to perform slightly better for image classification.
    """

    def __init__(self, channels, kernel_size, final_relu=False, bias=False, do_norm=True, norm='batch'):
        super().__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        self.padding = (self.kernel_size[0] - 1) // 2
        self.final_relu = final_relu

        norm_layer, relu_layer = norm_relu_layer(self.channels, do_norm, norm, relu=None)
        self.layers = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, self.kernel_size, padding=self.padding, bias=bias),
            norm_layer,
            nn.ReLU(),
            nn.Conv2d(self.channels, self.channels, self.kernel_size, padding=self.padding, bias=bias),
            norm_layer
        )

    def forward(self, input):
        # input (N x channels x H x W)
        # output (N x channels x H x W)
        out = self.layers(input)
        if self.final_relu:
            return nn.ReLU(out + input)
        else:
            return out + input

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

class Discriminator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, bias = False, norm = 'batch', sigmoid=True):
        super(Discriminator, self).__init__()

        self.sigmoid = sigmoid

        # 286x286 discriminator
        self.disc1 = EncoderBlock(in_channels * 2, 64, bias=bias, do_norm=False, do_activation=False)
        self.disc2 = EncoderBlock(64, 128, bias=bias, norm=norm)
        self.disc3 = EncoderBlock(128, 256, bias=bias, norm=norm)
        self.disc4 = EncoderBlock(256, 512, bias=bias, norm=norm)
        self.disc5 = EncoderBlock(512, 512, bias=bias, norm=norm)
        self.disc6 = EncoderBlock(512, 512, bias=bias, stride=1, norm=norm)
        self.disc7 = EncoderBlock(512, out_channels, bias=bias, stride=1, do_norm=False)
        self.pool = nn.AvgPool2d(6)

    def forward(self, x, ref):
        d1 = self.disc1(torch.cat([x, ref],1))
        d2 = self.disc2(d1)
        d3 = self.disc3(d2)
        d4 = self.disc4(d3)
        d5 = self.disc5(d4)
        d6 = self.disc6(d5)
        d7 = self.disc7(d6)
        d8 = self.pool(d7)
        if self.sigmoid:
            final = nn.Sigmoid()(d8)
        else:
            final = d7
        return final

class Generator(nn.Module):
    """
    The Generator architecture in < Perceptual Losses for Real-Time Style Transfer and Super-Resolution >
    by Justin Johnson, et al.
    """

    def __init__(self, in_channels=1, out_channels=1, do_norm=True, norm='batch', bias=True):
        super(Generator, self).__init__()
        model = []
        model += [Conv_Norm_ReLU(in_channels, 32, (7, 7), padding=3, stride=1, bias=bias, do_norm=do_norm, norm=norm),
                  # c7s1-32
                  Conv_Norm_ReLU(32, 64, (3, 3), padding=1, stride=2, bias=bias, do_norm=do_norm, norm=norm),  # d64
                  Conv_Norm_ReLU(64, 128, (3, 3), padding=1, stride=2, bias=bias, do_norm=do_norm, norm=norm)]  # d128
        for i in range(6):
            model += [ResidualLayer(128, (3, 3), final_relu=False, bias=bias)]  # R128
        model += [
            Deconv_Norm_ReLU(128, 64, (3, 3), padding=1, output_padding=1, stride=2, bias=bias, do_norm=do_norm, norm=norm),
            # u64
            Deconv_Norm_ReLU(64, 32, (3, 3), padding=1, output_padding=1, stride=2, bias=bias, do_norm=do_norm, norm=norm),

            Deconv_Norm_ReLU(32, 16, (3, 3), padding=1, output_padding=1, stride=2, bias=bias, do_norm=do_norm,
                             norm=norm),
            # u32
            nn.Conv2d(16, out_channels, (7, 7), padding=3, stride=1, bias=bias),  # c7s1-3
            nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """
        :param input: (N x channels x H x W)
        :return: output: (N x channels x H x W) with numbers of range [-1, 1] (since we use tanh())
        """
        return self.model(input)


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
generator = Generator(bias=False, norm="batch")
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
        validity = discriminator(gen_imgs, real_imgs)
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
