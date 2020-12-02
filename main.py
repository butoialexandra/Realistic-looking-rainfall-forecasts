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
from dataset import UnconditionalDataset
from torch.utils.tensorboard import SummaryWriter

# from src.unconditional.discriminator import DiscriminatorMy

from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np

datetimestr = datetime.now().strftime("%d-%b-%Y-%H:%M")
output_dir = f"ouput_{datetimestr}"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cosmo | cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=False, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
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

device = torch.device("cuda:1" if opt.cuda else "cpu")
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
    dataset = UnconditionalDataset(device=device, on_cluster=opt.on_cluster)
    nc = 1

elif opt.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                            transform=transforms.ToTensor())
    nc=3

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=False, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)

# # function to plot images
# def plot_images(images, path):
#     plt.figure(figsize=(16,16))
#     nrow = np.sqrt(images.shape[0])
#     for i in range(images.shape[0]):
#         plt.subplot(8,8,i+1)
#         plt.imshow(images[i][0], vmin=0, vmax=1)
#     plt.savefig(path)


def plot_images(gen_imgs, save_path):
    ndata = gen_imgs.shape[0]
    ncols = 5
    nrows = int(ndata/5) + 1
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(6*ncols,5*nrows))#, figsize=(5*ncols, 5))
    for i in range(gen_imgs.shape[0]):
        x = int(i % nrows)
        y = int(i / nrows)
        im = axs[x,y].pcolormesh(gen_imgs[i,0,:,:].cpu().data, cmap='viridis')
        fig.colorbar(im, ax=axs[x,y])
    # for i in range(real_imgs.shape[0]):
        # im = axs[1, i].pcolormesh(real_imgs[i,:,:].cpu().data, cmap='viridis')
        # fig.colorbar(im, ax=axs[1, i])
    plt.savefig(save_path)
    return fig


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


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
            nn.Sigmoid()
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
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
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
            noisy_encoded = torch.cat((encoded,noise),0)
            output = nn.parallel.data_parallel(self.decoded, noisy_encoded, range(self.ngpu))
        else:
            encoded = self.encoder(input)
            noisy_encoded = torch.cat((encoded, noise),0)
            output = self.decoder(noisy_encoded)
        return output
if opt.conditional:
    netG = CondGenerator(ngpu).to(device)
else:
    netG = Generator(ngpu).to(device)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        bias = False
        self.main = nn.Sequential(
            # input is (nc) x 128 x 192
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=bias),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 96
            nn.Conv2d(ndf, ndf, 4, 2, 1, bias=bias),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 48
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=bias),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 24
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=bias),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 12
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=bias),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 6
            nn.Conv2d(ndf * 8, 1, (4,6), 1, 0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()

fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr / 2, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr * 2, betas=(0.5, 0.999))

if opt.dry_run:
    opt.niter = 1

writer = SummaryWriter(f"{opt.outf}")
skip_d = False


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
        loss_D_real = criterion(output, label)
        # loss_D_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        fake_label = torch.zeros_like(label)
        output = netD(fake.detach())
        loss_D_fake = criterion(output, fake_label)
        # loss_D_fake.backward()
        D_G_z1 = output.mean().item()
        loss_D = loss_D_real + loss_D_fake

        if not skip_d:
            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(netD.parameters(), 1.0)
            optimizerD.step()
        skip_d = False

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        real_labels = torch.ones_like(label)
        # label.fill_(real_label)  # fake labels are real for generator cost
        fake = netG(noise)
        output = netD(fake)
        loss_G = criterion(output, real_labels)
        # loss_G = torch.mean((netD(netG(noise)) - torch.ones_like(label)) ** 2)
        loss_G.backward()
        torch.nn.utils.clip_grad_norm_(netG.parameters(), 1.0)
        D_G_z2 = output.mean().item()
        optimizerG.step()

        if loss_G > 10.0 and loss_D < 0.10:
            skip_d = True

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 loss_D.item(), loss_G.item(), D_x, D_G_z1, D_G_z2))
        writer.add_scalar('Discriminator loss', loss_D.item(), epoch * len(dataloader) + i)
        writer.add_scalar('Generator loss', loss_G.item(), epoch * len(dataloader) + i)

        if i == 0:#% 100 == 0:
            if i == 0:
                plot_images(real_cpu, f"{opt.outf}/real_samples.png")
            fake = netG(fixed_noise)
            fig = plot_images(fake.detach(), f"{opt.outf}/fake_samples_epoch_{epoch}.png")
            writer.add_figure('Generated images', fig, global_step=epoch * len(dataloader) + 1)
        if opt.dry_run:
            break
    # do checkpointing
    if epoch % 5 == 0:
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
