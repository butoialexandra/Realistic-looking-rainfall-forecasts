"""
Importing header files
"""

import argparse
import os
import random

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from dataset import *
import utils
from generator import *
from discriminator import *

#Output directory
datetimestr = datetime.now().strftime("%d-%b-%Y-%H:%M")
output_dir = f"output_{datetimestr}"

parser = argparse.ArgumentParser()

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
parser.add_argument('--on_cluster', default=False, action='store_true', help='flag to indicate if using cluster')
parser.add_argument('--conditional', default=False, action='store_true', help='flag to indicate if using conditional GAN')
parser.add_argument('--highres', default=False, action='store_true', help='flag to indicate if using high resolution observations')
opt = parser.parse_args()

print(opt)

#Setting parameters
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)

#Creating output directory
try:
    os.makedirs(opt.outf)
except OSError:
    pass

# Setting random seed
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

#Cuda benchmark
cudnn.benchmark = True

#Cuda availability
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

#Device
device = torch.device("cuda:1" if opt.cuda else "cpu")

#Loading dataset
if opt.conditional:
    dataset = ConditionalDataset(device=device, highres=opt.highres)
else:
    dataset = UnconditionalDataset(device=device, highres=opt.on_cluster)
nc = 1
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=False, num_workers=int(opt.workers))
#Generator
if opt.conditional and opt.highres:
    netG = CondGeneratorHighres(ngpu).to(device)
elif opt.conditional and not opt.highres:
    netG = CondGenerator(ngpu).to(device)
elif (not opt.conditional) and opt.highres:
    netG = GeneratorHighres(ngpu).to(device)
else:
    netG = Generator(ngpu).to(device)

netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)


#Discriminator
if opt.highres:
    netD = DiscriminatorHighres(ngpu).to(device)
else:
    netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

#Loss
criterion = nn.BCELoss()

#Noise vector
fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
#Real and fake
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr / 2, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr * 2, betas=(0.5, 0.999))

if opt.dry_run:
    opt.niter = 1

writer = SummaryWriter(f"{opt.outf}")
skip_d = False

if opt.conditional:                                                                                                                                                                                [85/1327]
    for epoch in range(opt.niter):
        for i, (x,y) in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            obs = y.to(device).float()
            netD.zero_grad()
            pred = x.to(device).float()

            batch_size = obs.size(0)
            label = torch.full((batch_size,), real_label,
                               dtype=real_cpu.dtype, device=device)
            output = netD(obs)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(pred,noise)
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
            writer.add_scalar("Discriminator loss", errD.item(), epoch)                                                                                                                            [38/1327]
            if i % 100 == 0:
                if epoch == 0:
                    fig = utils.plot_images_grid(obs.cpu().numpy(), f"{opt.outf}/real_samples_epoch_{epoch}.png")

                fake = netG(pred, fixed_noise)

                fig = utils.plot_images_grid(fake.detach().cpu().numpy(), f"{opt.outf}/fake_samples_epoch_{epoch}.png")
                writer.add_figure('Generated_images', fig, global_step=epoch*len(dataloader)+ 101)
                utils.plot_images_compare(fake[0][0].detach().cpu().numpy(), real_cpu[0][0].cpu().numpy(), f"{opt.outf}/image_pair_{epoch}.png")
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
                fig = plot_images_grid(real_cpu.cpu().numpy(), f"{opt.outf}/real_samples_epoch_{epoch}.png")
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

