import argparse
import numpy as np
import datetime
import matplotlib.pyplot as plt

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

from discriminator import Discriminator
from generator import Generator
from dataset import WGANDataset

def sample_image(generator, cuda, n_row, latent_dim, img_shape):
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row, latent_dim))))

    gen_imgs = generator(z, img_shape)
    # gen_imgs = gen_imgs.unsqueeze(1)

    # fig = plt.figure(figsize=(30, 3))
    # grid = torchvision.utils.make_grid(gen_imgs, nrow=n_row)
    # plt.imshow(grid.permute(1, 2, 0), cmap='viridis')
    # plt.colorbar(orientation='horizontal')
    fig, axs = plt.subplots(ncols=n_row, figsize=(5*n_row, 5))
    for i in range(gen_imgs.shape[0]):
        axs[i].pcolormesh(gen_imgs[i,:,:].data, cmap='viridis')
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_shape", type=int, default=(127, 188), help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
    parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between image samples")
    opt = parser.parse_args()
    print(opt)

    cuda = True if torch.cuda.is_available() else False
    on_cluster = False

    # Initialize generator and discriminator
    generator = Generator(opt.latent_dim, opt.img_shape)
    discriminator = Discriminator(opt.img_shape)

    if cuda:
        generator.cuda()
        discriminator.cuda()

    device = torch.device("cuda:0" if cuda else "cpu")

    training_params = {"batch_size": opt.batch_size, "shuffle": True, "num_workers": 0}
    data = WGANDataset(on_cluster=on_cluster, device=device)
    dataloader = torch.utils.data.DataLoader(data, **training_params)

    # Optimizers
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    writer = SummaryWriter('runs/{}'.format(datetime.datetime.now()))

    # ----------
    #  Training
    # ----------
    batches_done = 0

    print('Started Training')
    for epoch in range(opt.n_epochs):
        running_loss_G = 0.0
        running_loss_D = 0.0

        for i, imgs in enumerate(dataloader):

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            fake_imgs = generator(z, opt.img_shape).detach()
            # Adversarial loss
            loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))

            loss_D.backward()
            optimizer_D.step()
            running_loss_D += loss_D.item()

            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-opt.clip_value, opt.clip_value)

            # Train the generator every n_critic iterations
            if i % opt.n_critic == 0:
                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Generate a batch of images
                gen_imgs = generator(z, opt.img_shape)
                # Adversarial loss
                loss_G = -torch.mean(discriminator(gen_imgs))

                loss_G.backward()
                optimizer_G.step()
                running_loss_G += loss_D.item()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (
                    epoch, opt.n_epochs, batches_done % len(dataloader), len(dataloader), loss_D.item(), loss_G.item())
                )

            batches_done += 1

        writer.add_scalar('Discriminator loss', running_loss_D / len(dataloader), epoch)
        writer.add_scalar('Generator loss', running_loss_G / (len(dataloader) // opt.n_critic), epoch)
        writer.add_figure('Generated images', sample_image(generator, cuda, 10, opt.latent_dim, opt.img_shape),
                          global_step=epoch)

    print('Finished Training')
