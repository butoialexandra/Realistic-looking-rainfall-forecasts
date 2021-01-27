from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from dataset import UnconditionalDataset, ConditionalDataset, UnconditionalDatasetObservations
from torch.utils.tensorboard import SummaryWriter
import math

from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np

import time

datetimestr = datetime.now().strftime("%d-%b-%Y-%H:%M")
output_dir = f"ouput_{datetimestr}"
writer = SummaryWriter()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cosmo | cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=False, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=92, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--dry-run', action='store_true', help='check a single training cycle works')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default=output_dir, help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--classes', default='bedroom', help='comma separated list of classes for the lsun data set')
parser.add_argument('--on_cluster', default=False, action='store_true', help='flag to indicate if using cluster')
parser.add_argument('--conditional', default=False, action='store_true', help='flag to indicate if using cluster')
parser.add_argument('--highres', default=False, action='store_true', help='flag to indicate if using cluster')
parser.add_argument('--test_fraction', default=0.25, type=float, help='fraction of data to be used for testing for cGAN')
opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
  
if opt.dataroot is None and str(opt.dataset).lower() in ['imagenet', 'folder', 'lfw', 'lsun', 'cifar10', 'mnist']:
    raise ValueError("`dataroot` parameter is required for dataset \"%s\"" % opt.dataset)

device = torch.device("cuda:0" if opt.cuda else "cpu")
if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    nc=3
elif opt.dataset == 'lsun':
    classes = [ c + '_train' for c in opt.classes.split(',')]
    dataset = dset.LSUN(root=opt.dataroot, classes=classes,
                        transform=transforms.Compose([
                            transforms.Resize(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
    nc=3
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    nc=3

elif opt.dataset == 'mnist':
        dataset = dset.MNIST(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                           ]))
        nc=1


elif opt.dataset == 'cosmo':
    if opt.conditional:
        dataset = ConditionalDataset(device=device, img_size = opt.imageSize, highres = opt.highres)
    else:
        dataset = UnconditionalDatasetObservations(device=device, on_cluster=opt.on_cluster)
        print("Finished loading dataset")
    main_time = time.time()
    nc = 1

elif opt.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                            transform=transforms.ToTensor())
    nc=3

assert dataset
if opt.conditional:
    num_data = len(dataset)
    train_idx = range(math.floor((1.0 - opt.test_fraction)*num_data))
    test_idx = range(math.ceil((1.0 - opt.test_fraction)*num_data), num_data)
    train_dataset = dataset[train_idx]
    test_dataset = dataset[test_idx]
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = opt.batchSize,
                                                   shuffle=True, num_workers=int(opt.workers))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = opt.batchSize,
                                                   shuffle=False, num_workers=int(opt.workers))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=False, num_workers=int(opt.workers))
print("Finished Dataloader", time.time()-main_time)
main_time = time.time()

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)

if opt.conditional:
    nc_disc = 2*nc

print("Finished setting parameters", time.time()-main_time)
main_time = time.time()
# function to plot images
def plot_images(images, path):
    fig, axs =  plt.subplots(ncols=8, nrows=8, figsize=(16,16))
    #nrow = np.sqrt(images.shape[0])
    for i in range(images.shape[0]):
        row_num = int(i / 8)
        col_num = int(i % 8)
        im = axs[row_num, col_num].pcolormesh(images[i,0], vmin=0, vmax=1, cmap='viridis')
        fig.colorbar(im, ax=axs[row_num, col_num])
    plt.savefig(path)


#Function to plot single image pair -- conditional
def plot_image_single_conditional(generated, observed, path):
    plt.figure(figsize=(6,3.2))
    fig, ax  = plt.subplots(1,2)
    ax[0].set_title("Generated")
    ax[0].imshow(generated, vmin=0, vmax=1)
    ax[1].set_title("Observed")
    ax[1].imshow(observed, vmin=0, vmax=1)
    plt.savefig(path)

#Function to plot single image unconditional
def plot_image_single_unconditional(generated, path):
    plt.figure(figsize=(4,4))
    plt.imshow(generated, vmin=0, vmax=1)
    plt.savefig(path)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)
class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ListModule(torch.nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 16, (4,6), 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d(ngf * 4,     ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class GeneratorHighres(nn.Module):
    def __init__(self, ngpu):
        super(GeneratorHighres, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 32, (4,6), 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 32),
            nn.ReLU(True),
            # state size. (ngf*16) x 4 x 4
            nn.ConvTranspose2d(ngf * 32, ngf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d(ngf * 8,     ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(ngf * 4,     ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output
class CondGenerator(nn.Module):
    def __init__(self, ngpu):
        super(CondGenerator, self).__init__()
        self.ngpu = ngpu
        self.encoder = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, 1, 1, bias=False),
            nn.Flatten()
            #nn.BatchNorm2d(ndf * 2),
            #nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            #nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 4),
            #nn.LeakyReLU(0.2, inplace=True),
            ## state size. (ndf*4) x 8 x 8
            #nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 8),
            #nn.LeakyReLU(0.2, inplace=True),
            ## state size. (ndf*8) x 4 x 4
            #nn.Conv2d(ndf * 8, 1, (4,6), 1, 0, bias=False),
            #nn.LeakyReLU(0.2, inplace=True)
        )

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(384+nz, ngf * 16, (4,6), 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 4 x 6
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 12
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 24
            nn.ConvTranspose2d(ngf * 4,     ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 48
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 96
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 192
        )

    def forward(self, input, noise):
        if input.is_cuda and self.ngpu > 1:
            encoded = nn.parallel.data_parallel(self.encoder, input, range(self.ngpu))
            encoded = encoded.view(encoded.size(0), encoded.size(1), 1, 1)
            noisy_encoded = torch.cat((encoded,noise),1)
            output = nn.parallel.data_parallel(self.decoded, noisy_encoded, range(self.ngpu))
        else:
            encoded = self.encoder(input)
            encoded = encoded.view(encoded.size(0), encoded.size(1), 1, 1)
            noisy_encoded = torch.cat((encoded, noise),1)
            output = self.decoder(noisy_encoded)
        return output


class CondGeneratorHighres(nn.Module):
    def __init__(self, ngpu):
        super(CondGeneratorHighres, self).__init__()
        self.ngpu = ngpu
        self.encoder = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, 1, 1, bias=False),
            #add flatten layer
            nn.Flatten()
            #nn.BatchNorm2d(ndf * 2),
            #nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            #nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 4),
            #nn.LeakyReLU(0.2, inplace=True),
            ## state size. (ndf*4) x 8 x 8
            #nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 8),
            #nn.LeakyReLU(0.2, inplace=True),
            ## state size. (ndf*8) x 4 x 4
            #nn.Conv2d(ndf * 8, 1, (4,6), 1, 0, bias=False),
            #nn.LeakyReLU(0.2, inplace=True)
        )

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(384+nz, ngf * 32, (4,6), 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 32),
            nn.ReLU(True),
            # state size. (ngf*16) x 4 x 6
            nn.ConvTranspose2d(ngf * 32, ngf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 12
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 24
            nn.ConvTranspose2d(ngf * 8,     ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 48
            nn.ConvTranspose2d(ngf * 4,     ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 96
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 96
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 192
        )

    def forward(self, input, noise):
        if input.is_cuda and self.ngpu > 1:
            encoded = nn.parallel.data_parallel(self.encoder, input, range(self.ngpu))
            encoded = encoded.view(encoded.size(0),encoded.size(1),1,1)
            noisy_encoded = torch.cat((encoded,noise),1)
            output = nn.parallel.data_parallel(self.decoded, noisy_encoded, range(self.ngpu))
        else:
            encoded = self.encoder(input)
            encoded = encoded.view(encoded.size(0), encoded.size(1),1,1)
            noisy_encoded = torch.cat((encoded, noise),1)
            output = self.decoder(noisy_encoded)
        return output

class CondGeneratorHighresTile(nn.Module):
    def __init__(self, ngpu):
        super(CondGeneratorHighresTile, self).__init__()
        self.ngpu = ngpu
        self.encoder = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 8 x 8
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, 1, 1, bias=False),
            #add flatten layer
            nn.Flatten()
            #nn.BatchNorm2d(ndf * 2),
            #nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            #nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 4),
            #nn.LeakyReLU(0.2, inplace=True),
            ## state size. (ndf*4) x 8 x 8
            #nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 8),
            #nn.LeakyReLU(0.2, inplace=True),
            ## state size. (ndf*8) x 4 x 4
            #nn.Conv2d(ndf * 8, 1, (4,6), 1, 0, bias=False),
            #nn.LeakyReLU(0.2, inplace=True)
        )

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(16+nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*16) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(ngf,     nc, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input, noise):
        if input.is_cuda and self.ngpu > 1:
            encoded = nn.parallel.data_parallel(self.encoder, input, range(self.ngpu))
            encoded = encoded.view(encoded.size(0),encoded.size(1),1,1)
            noisy_encoded = torch.cat((encoded,noise),1)
            output = nn.parallel.data_parallel(self.decoded, noisy_encoded, range(self.ngpu))
        else:
            encoded = self.encoder(input)
            encoded = encoded.view(encoded.size(0), encoded.size(1),1,1)
            noisy_encoded = torch.cat((encoded, noise),1)
            output = self.decoder(noisy_encoded)
        return output



class CondGeneratorHighresBig(nn.Module):

    def __init__(self, ngpu, n_filters=4, skip_connections=False):
        super(CondGeneratorHighresBig, self).__init__()
        self.ngpu = ngpu
        #self.input_dim = input_dim
        self.n_filters = n_filters
        self.use_skip_connections = skip_connections    

        self.encoder = ListModule(
            nn.Sequential(
                # input is (nc) x 128 x 192
                nn.Conv2d(1, n_filters, 4, 2, 1, bias=False),
                nn.BatchNorm2d(n_filters),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                # n_filters x 64 x 96
                nn.Conv2d(n_filters, n_filters, 4, 2, 1, bias=False),
                nn.BatchNorm2d(n_filters),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                # n_filters x 32 x 48
                nn.Conv2d(n_filters, n_filters * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(n_filters * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                # n_filters x 16 x 24
                nn.Conv2d(n_filters * 2, n_filters * 2, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # output: also 1 x 16 x 24 = 384
            ),
            nn.Sequential(
                # n_filters x 8 x 12
                nn.Conv2d(n_filters * 2, 1, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # output: also 1 x 8 x 12 = 96
            )
        )

        decoder_filters = [
            n_filters * 32,
            n_filters * 16,
            n_filters * 8,
            n_filters * 4,
            n_filters * 2,
            n_filters,
            1
        ]
        
        decoder_input_channels = [
            96,
            n_filters * 32,
            n_filters * 16,
            n_filters * 8,
            n_filters * 4,
            n_filters * 2,
            n_filters
        ]

        if self.use_skip_connections:
            decoder_input_channels[2] += n_filters * 2
            decoder_input_channels[3] += n_filters * 2
            decoder_input_channels[4] += n_filters
            decoder_input_channels[5] += n_filters
            decoder_input_channels[6] += 1

        decoder_layers = []
        use_bias = True
        # Generate N blocks of deconvolution-batchnorm-nonlinearity
        for i, (in_channels, out_channels) in enumerate(zip(decoder_input_channels, decoder_filters)):
            block_layers = []
            if i == 0:
                # first layer has different parameters
                kernel_size = (4,6)
                stride, padding = 1, 0
            else:
                kernel_size, stride, padding = 4, 2, 1
            block_layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=use_bias))
            if out_channels > 1:  # (i < len(decoder_filters) - 1):
                block_layers.append(nn.BatchNorm2d(out_channels))
                block_layers.append(nn.ReLU(inplace=True))
            else:
                # last layer can't have batchnorm, because it would output all zeros
                block_layers.append(nn.Tanh())
            decoder_layers.append(nn.Sequential(*block_layers))

        self.decoder = ListModule(*decoder_layers)

        #self.decoder_blocks = ListModule(
        #    nn.Sequential(
        #        # (384 = output of encoder) + (input_dim = 1,2)
        #        nn.ConvTranspose2d(96, n_filters * 32, (4,6), 1, 0, bias=False),
        #        nn.BatchNorm2d(n_filters * 32),
        #        nn.ReLU(True),
        #    ),
        #    nn.Sequential(
        #        # (n_filters*16) x 4 x 6
        #        nn.ConvTranspose2d(n_filters * 32, n_filters * 16, 4, 2, 1, bias=False),
        #        nn.BatchNorm2d(n_filters * 16),
        #        nn.ReLU(True),
        #    ),
        #    nn.Sequential(
        #        # (n_filters*8) x 8 x 12
        #        nn.ConvTranspose2d(n_filters * 16 + n_filters * 2, n_filters * 8, 4, 2, 1, bias=False),
        #        nn.BatchNorm2d(n_filters * 8),
        #        nn.ReLU(True),
        #    ),
        #    nn.Sequential(
        #        # (n_filters*4) x 16 x 24
        #        nn.ConvTranspose2d(n_filters * 8 + n_filters * 2, n_filters*4, 4, 2, 1, bias=False),
        #        nn.BatchNorm2d(n_filters * 4),
        #        nn.ReLU(True),
        #    ),
        #    nn.Sequential(
        #        # (n_filters*2) x 32 x 48
        #        nn.ConvTranspose2d(n_filters * 4 + n_filters,n_filters * 2, 4, 2, 1, bias=False),
        #        nn.BatchNorm2d(n_filters*2),
        #        nn.ReLU(True),
        #    ),
        #    nn.Sequential(
        #        # (n_filters) x 64 x 96
        #        nn.ConvTranspose2d(n_filters * 2 + n_filters, n_filters, 4, 2, 1, bias=False),
        #        nn.BatchNorm2d(n_filters),
        #        nn.ReLU(True),
        #    ),
        #    nn.Sequential(
        #        # (n_filters) x 128 x 192
        #        nn.ConvTranspose2d(n_filters + 1, 1, 4, 2, 1, bias=False),
        #        nn.Tanh()
        #        # output: 1 x 256 x 384
        #    )
        #)

    def forward(self, input, noise):
        #noise = noise.unsqueeze(-1).unsqueeze(-1)

        # if input.is_cuda and self.ngpu > 1:
            # encoded = nn.parallel.data_parallel(self.encoder, input, range(self.ngpu))
            # encoded = encoded.view(encoded.size(0),encoded.size(1),1,1)
            # noisy_encoded = torch.cat((encoded,noise),1)
            # output = nn.parallel.data_parallel(self.decoded, noisy_encoded, range(self.ngpu))
        # else:
            # encoded = self.encoder(input)
            # encoded = encoded.view(encoded.size(0), encoded.size(1),1,1)
            # noisy_encoded = torch.cat((encoded, noise),1)
            # output = self.decoder(noisy_encoded)

        # TODO(mzilinec): fix for multiple GPUs
        
        output = self._forward(input, noise)
        return output

    def _forward(self, cond_input, noise):
        skip_conn_tgts = [2, 3, 4, 5, 6]
        skip_conns = []
        x = cond_input

        for block in self.encoder:
            # create skip connection to decoder
            if self.use_skip_connections:
                skip_conns.append(x)
            # apply encoder block
            x = block(x)

        x = torch.flatten(x, start_dim=1)
        x = x.view(x.size(0), x.size(1), 1, 1)
        x = torch.cat((x, noise), dim=1)  # TODO(mzilinec): i DISABLED noise (see pix2pix)
        # TODO(mzilinec): the 2nd /noise/ dim likely gets ignored, we should *add* them instead

        for i, block in enumerate(self.decoder):
            # apply skip connection from encoder
            if self.use_skip_connections and i == skip_conn_tgts[0]:
                skip_conn_tgts = skip_conn_tgts[1:]
                skip_conn = skip_conns.pop()
                x = torch.cat((x, skip_conn), dim=1)
                # print(x.size())
            # apply decoder block
            x = block(x)
        
        return x



#add condition for highres
if opt.conditional and opt.highres:
    netG = CondGeneratorHighresBig(ngpu).to(device)
elif opt.conditional and not opt.highres:
    netG = CondGenerator(ngpu).to(device)
elif opt.highres:
    netG = GeneratorHighres(ngpu).to(device)
else:
    netG = Generator(ngpu).to(device)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)


print("Finished setting generators",time.time()-main_time)
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc_disc) x 128 x 192
            nn.Conv2d(nc_disc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 96
            nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 48
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 24
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 12
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 6
            nn.Conv2d(ndf * 8, 1, (4,6), 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


class DiscriminatorHighres(nn.Module):
    def __init__(self, ngpu):
        super(DiscriminatorHighres, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc_disc) x 128 x 192
            nn.Conv2d(nc_disc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 96
            nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 48
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 24
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 12
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 6
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 6
            nn.Conv2d(ndf * 16, 1, (4,6), 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)

class DiscriminatorHighresTile(nn.Module):
    def __init__(self, ngpu):
        super(DiscriminatorHighresTile, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc_disc) x 64 x 64
            nn.Conv2d(nc_disc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            # state size. (ndf*8) x 4 x 6
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)
if opt.highres:
    netD = DiscriminatorHighresTile(ngpu).to(device)
else:
    netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()
# Fix a seed so that we get same results every time
fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


def test_loop_conditional(netG):
    crps = 0
    lsd = 0
    rmse = 0
    step = 0
    for i, (x, y) in enumerate(valid_ds, 0):
        step += 1
        data = y
        real = data.to(device)
        pred = x.to(device)
        real = real.float()
        pred = pred.float()
        batch_size = real.size(0)

        noise = torch.randn(batch_size, nz, device=device)
        fake = netG(pred, noise)

        real = real.detach().cpu().numpy()
        fake = fake.detach().cpu().numpy()
        lsd += log_spectral_distance_pairs_avg(real, fake)
        rmse += rmse(real, fake)
        ensemble = generate_ensemble(netG, pred, 10)
        ensemble = ensemble.detach().cpu().numpy()
        crps += crps_ensemble(real, ensemble)

    lsd, rmse, crps = lsd / step, rmse / step, crps / step
    return lsd, rmse, crps


def generate_ensemble(netG, pred, no_members):
    batch_size = pred.size(0)
    noise = torch.randn(batch_size, nz, device=device)
    fake = netG(pred, noise)
    fake = torch.unsqueeze(fake, 2)

    for i in range(no_members - 1):
        noise = torch.randn(batch_size, nz, device=device)
        new_fake = netG(pred, noise)
        new_fake = torch.unsqueeze(new_fake, 2)
        fake = torch.cat([fake, new_fake], 2)

    return fake




if opt.dry_run:
    opt.niter = 1
if opt.conditional:
    for epoch in range(opt.niter):
        for i, (x,y) in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            data = y
            netD.zero_grad()
            real_cpu = data.to(device)
            pred = x.to(device)
            real_cpu = real_cpu.float()
            pred = pred.float()
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label,
                               dtype=real_cpu.dtype, device=device)
            if opt.highres:
                pred_double = torch.repeat_interleave(pred, 2, dim=2)
                pred_double = torch.repeat_interleave(pred_double, 2, dim=3)
                ip_disc = torch.cat((real_cpu, pred_double), 1)
                ip_disc = ip_disc.float()
            else:
                ip_disc = torch.cat((real_cpu, pred), 1)
                ip_disc = ip_disc.float()
            output = netD(ip_disc)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            print(f"Pred shape: {pred.shape}, Noise shape: {noise.shape}")
            fake = netG(pred,noise)
            label.fill_(fake_label)
            fake_nograd = fake.detach()
            if opt.highres:
                #pred_double = torch.repeat_interleave(pred, 2, dim=2)
                #pred_double = torch.repeat_interleave(pred_double, 2, dim=3)
                ip_disc_fake = torch.cat((fake_nograd, pred_double), 1)
                ip_disc_fake = ip_disc_fake.float()
            else:
                ip_disc_fake = torch.cat((fake_nograd, pred), 1)
                ip_disc_fake = ip_disc_fake.float()
            output = netD(ip_disc_fake)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            if opt.highres:
                #pred_double = torch.repeat_interleave(pred, 2, dim=2)
                #pred_double = torch.repeat_interleave(pred_double, 2, dim=3)
                ip_gen = torch.cat((fake, pred_double), 1)
                ip_gen = ip_gen.float()
            else:
                ip_gen = torch.stack((fake, pred), dim=1)
                ip_gen = ip_gen.float()
            output = netD(ip_gen)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            if epoch%2 == 0:
                optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, opt.niter, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            writer.add_scalar("Generator loss", errG.item(), epoch)
            writer.add_scalar("Discriminator loss", errD.item(), epoch)
            if i % 100 == 0:
                #vutils.save_image(real_cpu,
                #        '%s/real_samples.png' % opt.outf,
                #        normalize=True)
                plot_images(real_cpu.cpu().numpy(), f"{opt.outf}/real_samples_epoch_{epoch}_batch_{i}.png")
                if opt.highres:
                    plot_images(pred_double.cpu().numpy(), f"{opt.outf}/pred_samples_epoch_{epoch}_batch_{i}.png")
                else:
                    plot_images(pred.cpu().numpy(), f"{opt.outf}/pred_samples_epoch_{epoch}_batch_{i}.png")
                fake = netG(pred, fixed_noise)
                #vutils.save_image(fake.detach(),
                #        '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                #        normalize=True)
                plot_images(fake.detach().cpu().numpy(), f"{opt.outf}/fake_samples_epoch_{epoch}_batch_{i}.png")
                plot_image_single_conditional(fake[0][0].detach().cpu().numpy(), real_cpu[0][0].cpu().numpy(), f"{opt.outf}/image_pair_{epoch}_batch_{i}.png")
            if opt.dry_run:
                break
        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
        writer.flush()
else:
    for epoch in range(opt.niter):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            if opt.dataset == 'cosmo':
                real_cpu = data.to(device)
            else:
                real_cpu = data[0].to(device)
            real_cpu = real_cpu.float()
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label,
                               dtype=real_cpu.dtype, device=device)
            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, opt.niter, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            writer.add_scalar("Generator loss", errG.item(), epoch)
            writer.add_scalar("Discriminator loss", errD.item(), epoch)
            if i % 100 == 0:
                #vutils.save_image(real_cpu,
                #        '%s/real_samples.png' % opt.outf,
                #        normalize=True)
                plot_images(real_cpu.cpu().numpy(), f"{opt.outf}/real_samples_epoch_{epoch}.png")
                fake = netG(fixed_noise)
                #vutils.save_image(fake.detach(),
                #        '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                #        normalize=True)
                plot_images(fake.detach().cpu().numpy(), f"{opt.outf}/fake_samples_epoch_{epoch}.png")
                plot_image_single_unconditional(fake[0][0].detach().cpu().numpy(), f"{opt.outf}/fake_image_epoch_{epoch}.png")
            if opt.dry_run:
                break
        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
        writer.flush()
writer.close()
