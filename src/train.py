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

from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np

import time
import socket

from discriminator import *
from generator import *
from utils import *


datetimestr = datetime.now().strftime("%d-%b-%Y-%H:%M")
output_dir = f"out_{datetimestr}"
writer = SummaryWriter()
on_cluster = socket.gethostname().endswith(".ethz.ch")
print(f"Running on cluster: {on_cluster}")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='cosmo', help='cosmo | cond-cosmo')
parser.add_argument('--model', required=False, default='gan', help='gan | cgan | unet')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--device', type=int, default=0, help='selects cuda device')
parser.add_argument('--dry-run', action='store_true', help='check a single training cycle works')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default=output_dir, help='folder to output images and model checkpoints')
parser.add_argument('--manual-seed', type=int, help='manual seed')
parser.add_argument('--conditional', default=False, action='store_true', help='flag to indicate if using cluster')
parser.add_argument('--highres', default=False, action='store_true', help='flag to indicate if using cluster')
parser.add_argument('--add-l1', default=False, action='store_true', help='whether to use l1 loss in addition to discriminator loss, as in pix2pix')
parser.add_argument('--add-skip-connections', default=False, action='store_true', help='whether to use skip connections in GAN, as in pix2pix')
opt = parser.parse_args()
print(opt)


try:
    os.makedirs(opt.outf)
except OSError:
    pass


if opt.manual_seed is None:
    opt.manual_seed = random.randint(1, 10000)
print("Random Seed: ", opt.manual_seed)
random.seed(opt.manual_seed)
torch.manual_seed(opt.manual_seed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device(f"cuda:{opt.device}" if opt.cuda else "cpu")


if opt.dataset == 'cosmo':
    if opt.model == 'gan':
        dataset = UnconditionalDatasetObservations(device=device, on_cluster=on_cluster)
    else:
        dataset = ConditionalDataset(device=device, highres = True)
    print("Finished loading dataset")
    main_time = time.time()
    nc = 1

elif opt.dataset == 'cond-cosmo':
    raise Exception("Not implemented")

else:
    raise Exception("Invalid dataset %s" % opt.dataset)


if opt.model in ['vae', 'cgan']:
    test_len = int(len(dataset) / 10)
    train_len = len(dataset) - test_len
    train_ds, valid_ds = torch.utils.data.random_split(dataset, lengths=[train_len, test_len], generator=torch.Generator().manual_seed(42))
    train_ds = torch.utils.data.DataLoader(train_ds, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
    valid_ds = torch.utils.data.DataLoader(valid_ds, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers))
else:
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))


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


# #add condition for highres
# if opt.conditional and opt.highres:
#     netG = CondGeneratorHighres(ngpu).to(device)
# elif opt.conditional and not opt.highres:
#     netG = CondGenerator(ngpu).to(device)
# elif opt.highres:
#     netG = GeneratorHighres(ngpu, nz).to(device)
# else:
#     netG = Generator(ngpu, nz).to(device)
# netG.apply(weights_init)
# if opt.netG != '':
#     netG.load_state_dict(torch.load(opt.netG))
# print(netG)

# if opt.highres:
#     netD = DiscriminatorHighres(ngpu, n_channels=1).to(device)
# else:
#     netD = Discriminator(ngpu, n_channels=1).to(device)
# netD.apply(weights_init)
# if opt.netD != '':
#     netD.load_state_dict(torch.load(opt.netD))
# print(netD)


class VAE(torch.nn.Module):

    def __init__(self, n_filters=64):
        super().__init__()

        self.encoder = nn.Sequential(
            # input is (nc) x 128 x 192
            nn.Conv2d(1, n_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x ?
            nn.Conv2d(n_filters, n_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x ?
            nn.Conv2d(n_filters, n_filters * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_filters*2, 1, 1, bias=False),
            nn.Flatten()
        )
    
        self.decoder = nn.Sequential(
            # 384 is the output of encoder AND input_dim is the noise input
            nn.ConvTranspose2d(384, n_filters * 32, (4,6), 1, 0, bias=False),
            nn.BatchNorm2d(n_filters * 32),
            nn.ReLU(True),
            # state size. (n_filters*16) x 4 x 6
            nn.ConvTranspose2d(n_filters * 32, n_filters * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters * 16),
            nn.ReLU(True),
            # state size. (n_filters*8) x 8 x 12
            nn.ConvTranspose2d(n_filters * 16, n_filters * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters * 8),
            nn.ReLU(True),
            # state size. (n_filters*4) x 16 x 24
            nn.ConvTranspose2d(n_filters * 8,     n_filters*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters*4),
            nn.ReLU(True),
            # state size. (n_filters*2) x 32 x 48
            nn.ConvTranspose2d(n_filters * 4,     n_filters*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters*2),
            nn.ReLU(True),
            # state size. (n_filters) x 64 x 96
            nn.ConvTranspose2d(n_filters * 2,     n_filters, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(True),
            # state size. (n_filters) x 64 x 96
            nn.ConvTranspose2d(n_filters, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 192
        )

    def forward(self, x):
        print("VAE Input:", x.size())
        latent = self.encoder(x)
        print("VAE latent:", latent.size())
        y_ = self.decoder(latent.unsqueeze(-1).unsqueeze(-1))
        print("VAE Output:", y_.size())
        return y_


def train_loop_conditional(opt, netG, netD, optimizerG, optimizerD, criterion, fixed_noise):
    global_step = 0
    if opt.add_l1:
        l1_loss = torch.nn.L1Loss(reduction='mean')

    for epoch in range(opt.epochs):
        for i, (x,y) in enumerate(train_ds, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            data = y
            netD.zero_grad()
            real_cpu = data.to(device).float()
            pred = x.to(device).float()
            batch_size = real_cpu.size(0)
            
            pred_double = upscale_input(pred)
            ip_disc = torch.cat((real_cpu, pred_double), 1)
            ip_disc = ip_disc.float()

            output = netD(ip_disc)
            errD_real = criterion(output, torch.ones((batch_size,), dtype=real_cpu.dtype, device=device))
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, nz, device=device)
            fake = netG(pred, noise)
            fake_nograd = fake.detach()
            
            # TODO
            ip_disc_fake = torch.cat((fake_nograd, pred_double), 1).float()

            output = netD(ip_disc_fake)
            errD_fake = criterion(output, torch.zeros((batch_size,), dtype=real_cpu.dtype, device=device))
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()

            ip_gen = torch.cat((fake, pred_double), 1).float()

            output = netD(ip_gen)
            errG = criterion(output, torch.ones((batch_size,), dtype=real_cpu.dtype, device=device))
            if opt.add_l1:
                errG += l1_loss(fake, real_cpu) * 100  # lambda=100 as in pix2pix paper
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, opt.epochs, i, len(train_ds),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            writer.add_scalar("Generator loss", errG.item(), global_step)
            writer.add_scalar("Discriminator loss", errD.item(), global_step)
            global_step += 1

            if i == 0: #len(dataloader) - 1:
                x, y = next(iter(valid_ds))
                x = x.to(device).float()
                y = y.to(device).float()
                y_ = netG(x, noise)
                # plot_images(real_cpu.cpu().numpy(), f"{opt.outf}/real_samples_epoch_{epoch}.png")
                # if opt.highres:
                    # plot_images(pred_double.cpu().numpy(), f"{opt.outf}/pred_samples_epoch_{epoch}.png")
                # else:
                    # plot_images(pred.cpu().numpy(), f"{opt.outf}/pred_samples_epoch_{epoch}.png")
                # plot_images(fake.detach().cpu().numpy(), f"{opt.outf}/fake_samples_epoch_{epoch}.png")
                plot_image_single_conditional(fake[0][0].detach().cpu().numpy(), real_cpu[0][0].cpu().numpy(), f"{opt.outf}/image_pair_{epoch}.png")
                plot_images_ncols(
                    x.cpu().squeeze(1), y.cpu().squeeze(1), y_.detach().cpu().squeeze(1), 
                    path=f"{opt.outf}/generated_{epoch}.png")
            if opt.dry_run:
                break
        # do checkpointing
        if epoch % 10 == 0 or epoch == opt.epochs - 1:
            torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
            torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
        writer.flush()
    writer.close()


def train_loop_unconditional(opt, netG, netD, optimizerG, optimizerD, criterion, fixed_noise):
    global_step = 0

    for epoch in range(opt.epochs):
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
            label = torch.full((batch_size,), 1.0,
                               dtype=real_cpu.dtype, device=device)
            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, nz, device=device)
            fake = netG(noise)
            label.fill_(0.0)
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
            label.fill_(1.0)  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, opt.epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            writer.add_scalar("Generator loss", errG.item(), global_step)
            writer.add_scalar("Discriminator loss", errD.item(), global_step)
            global_step += 1

            if i == len(dataloader) - 1:
                plot_images(real_cpu.cpu().numpy(), f"{opt.outf}/real_samples_epoch_{epoch}.png")
                fake = netG(fixed_noise)
                plot_images(fake.detach().cpu().numpy(), f"{opt.outf}/fake_samples_epoch_{epoch}.png")
                plot_image_single_unconditional(fake[0][0].detach().cpu().numpy(), f"{opt.outf}/fake_image_epoch_{epoch}.png")
            if opt.dry_run:
                break
        # do checkpointing
        if epoch % 10 == 0 or epoch == opt.epochs - 1:
            torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
            torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
        writer.flush()
    writer.close()


def train_loop_vae(opt, model, optimizer, criterion):
    global_step = 0

    for epoch in range(opt.epochs):
        for i, (x,y) in enumerate(train_ds, 0):
            model.zero_grad()
            y = y.to(device).float()
            x = x.to(device).float()

            batch_size = y.size(0)
            # pred_double = upscale_input(pred)
            # ip_disc = torch.cat((real_cpu, pred_double), 1)
            # print(ip_disc.dtype)
            # ip_disc = ip_disc.float()

            y_ = model(x)
            loss_ = criterion(y_, y)
            loss_.backward()
            optimizer.step()

            print('[%d/%d][%d/%d] Loss: %.4f'
                  % (epoch, opt.epochs, i, len(train_ds),
                     loss_.item()))
            writer.add_scalar("VAE MSE Loss", loss_.item(), global_step)
            global_step += 1

            if i == 0: #len(dataloader) - 1:
                x, y = next(iter(valid_ds))
                plot_images(y.cpu().numpy(), f"{opt.outf}/y_{epoch}.png")
                plot_images(x.cpu().numpy(), f"{opt.outf}/x_{epoch}.png")
                y_ = model(x.to(device).float())
                plot_images(y_.detach().cpu().numpy(), f"{opt.outf}/gen_{epoch}.png")
                plot_image_single_conditional(y_[0][0].detach().cpu().numpy(), y[0][0].cpu().numpy(), f"{opt.outf}/image_pair_{epoch}.png")
            if opt.dry_run:
                break
        # do checkpointing
        torch.save(model.state_dict(), '%s/vae_epoch_%d.pth' % (opt.outf, epoch))
        writer.flush()
    writer.close()


if opt.model in ['gan', 'cgan']:
    # if opt.conditional:
        # raise
    # else:
    if opt.model == 'gan':
        netG = GeneratorHighres(ngpu, nz).to(device)
    else:
        netG = CondGeneratorHighres(ngpu, nz, n_filters=64, skip_connections=opt.add_skip_connections).to(device)
    netG.apply(weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)
    print("Finished setting generators",time.time()-main_time)

    channels = 1 if opt.model == 'gan' else 2
    netD = DiscriminatorHighres(ngpu, n_channels=channels).to(device)
    netD.apply(weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    criterion = nn.BCELoss()
    # Fix a seed so that we get same results every time
    fixed_noise = torch.randn(opt.batchSize, nz, device=device)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr / 2, betas=(opt.beta1, 0.999))

    if opt.dry_run:
        opt.epochs = 1
    if opt.model == 'cgan':
        train_loop_conditional(opt, netG, netD, optimizerG, optimizerD, criterion, fixed_noise)
    else: # opt.model == 'gan':
        train_loop_unconditional(opt, netG, netD, optimizerG, optimizerD, criterion, fixed_noise)

elif opt.model == 'unet':
    pass

elif opt.model == 'vae':
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    criterion = torch.nn.MSELoss(reduction='sum')
    train_loop_vae(opt, model, optimizer, criterion)
