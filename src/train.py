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
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train for')
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
    if opt.conditional:
        dataset = ConditionalDataset(device=device, highres = opt.highres)
    else:
        dataset = UnconditionalDatasetObservations(device=device, on_cluster=on_cluster)
        print("Finished loading dataset")
    main_time = time.time()
    nc = 1

elif opt.dataset == 'cond-cosmo':
    raise Exception("Not implemented")

else:
    raise Exception("Invalid dataset %s" % opt.dataset)


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


if opt.conditional:
    raise
else:
    netG = GeneratorHighres(ngpu, nz).to(device)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

print("Finished setting generators",time.time()-main_time)

# if opt.highres:
#     netD = DiscriminatorHighres(ngpu, n_channels=1).to(device)
# else:
#     netD = Discriminator(ngpu, n_channels=1).to(device)
# netD.apply(weights_init)
# if opt.netD != '':
#     netD.load_state_dict(torch.load(opt.netD))
# print(netD)

netD = DiscriminatorHighres(ngpu, n_channels=1).to(device)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)


criterion = nn.BCELoss()
# Fix a seed so that we get same results every time
fixed_noise = torch.randn(opt.batchSize, nz, device=device)
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


def train_loop_conditional(opt):
    global_step = 0

    for epoch in range(opt.epochs):
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
            # label = torch.full((batch_size,), real_label,
                            #    dtype=real_cpu.dtype, device=device)
            # if opt.highres:
            #     pred_double = torch.repeat_interleave(pred, 2, dim=2)
            #     pred_double = torch.repeat_interleave(pred_double, 2, dim=3)
            #     ip_disc = torch.cat((real_cpu, pred_double), 1)
            #     ip_disc = ip_disc.float()
            # else:
            #     ip_disc = torch.cat((real_cpu, pred), 1)
            #     ip_disc = ip_disc.float()
            
            ip_disc = torch.cat((real_cpu, pred_double), 1)
            ip_disc = ip_disc.float()

            output = netD(ip_disc)
            errD_real = criterion(output, torch.ones((batch_size,), dtype=real_cpu.dtype, device=device))
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, nz, device=device)
            fake = netG(pred,noise)
            # label.fill_(fake_label)
            fake_nograd = fake.detach()
            # if opt.highres:
            #     ip_disc_fake = torch.cat((fake_nograd, pred_double), 1)
            #     ip_disc_fake = ip_disc_fake.float()
            # else:
            #     ip_disc_fake = torch.cat((fake_nograd, pred), 1)
            #     ip_disc_fake = ip_disc_fake.float()
            
            # TODO
            ip_disc_fake = torch.cat((fake_nograd, pred_double), 1)
            ip_disc_fake = ip_disc_fake.float()

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
            # label.fill_(real_label)  # fake labels are real for generator cost
            # if opt.highres:
            #     #pred_double = torch.repeat_interleave(pred, 2, dim=2)
            #     #pred_double = torch.repeat_interleave(pred_double, 2, dim=3)
            #     ip_gen = torch.cat((fake, pred_double), 1)
            #     ip_gen = ip_gen.float()
            # else:
            #     ip_gen = torch.stack((fake, pred), dim=1)
            #     ip_gen = ip_gen.float()

            ip_gen = torch.cat((fake, pred_double), 1)
            ip_gen = ip_gen.float()

            output = netD(ip_gen)
            errG = criterion(output, torch.ones((batch_size,), dtype=real_cpu.dtype, device=device))
            errG.backward()
            D_G_z2 = output.mean().item()
            if epoch%2 == 0:
                optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, opt.epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            writer.add_scalar("Generator loss", errG.item(), global_step)
            writer.add_scalar("Discriminator loss", errD.item(), global_step)
            global_step += 1

            if i % 100 == 0:
                plot_images(real_cpu.cpu().numpy(), f"{opt.outf}/real_samples_epoch_{epoch}.png")
                if opt.highres:
                    plot_images(pred_double.cpu().numpy(), f"{opt.outf}/pred_samples_epoch_{epoch}.png")
                else:
                    plot_images(pred.cpu().numpy(), f"{opt.outf}/pred_samples_epoch_{epoch}.png")
                fake = netG(pred, fixed_noise)
                plot_images(fake.detach().cpu().numpy(), f"{opt.outf}/fake_samples_epoch_{epoch}.png")
                plot_image_single_conditional(fake[0][0].detach().cpu().numpy(), real_cpu[0][0].cpu().numpy(), f"{opt.outf}/image_pair_{epoch}.png")
            if opt.dry_run:
                break
        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
        writer.flush()
    writer.close()


def train_loop_unconditional(opt):
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
            label = torch.full((batch_size,), real_label,
                               dtype=real_cpu.dtype, device=device)
            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, nz, device=device)
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
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
        writer.flush()
    writer.close()


if opt.dry_run:
    opt.epochs = 1
if opt.conditional:
    train_loop_conditional(opt)
else:
    train_loop_unconditional(opt)
