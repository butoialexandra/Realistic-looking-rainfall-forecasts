import argparse
import os
import datetime
import warnings
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import torch.autograd as autograd

from cond_dataset import Dataset
from dataset import ConditionalDataset
from src.modules import Generator, Discriminator, ESRGAN
from util import init_weights, plot_images
from utils import plot_power_spectrum
from verification import power_spectrum_batch_avg, power_spectrum_dB, log_spectral_distance_pairs_avg

# def calc_gradient_penalty(discriminator, real_data, fake_data, pred_data, batch_size, use_cuda, gpu, lmbda):
#     #print real_data.size()
#     alpha = torch.rand(batch_size, 1, 1, 1)
#     alpha = alpha.expand_as(real_data)
#     alpha = alpha.cuda(gpu) if use_cuda else alpha
#
#     interpolates = alpha * real_data + ((1 - alpha) * fake_data)
#
#     if use_cuda:
#         interpolates = interpolates.cuda(gpu)
#     interpolates = autograd.Variable(interpolates, requires_grad=True)
#
#     disc_interpolates = discriminator(interpolates, pred_data)
#
#     gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
#                               grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
#                                   disc_interpolates.size()),
#                               create_graph=True, retain_graph=True, only_inputs=True)[0]
#
#     gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lmbda
#     return gradient_penalty

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--sample_interval", type=int, default=1, help="interval between image sampling")
    parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
    parser.add_argument("--clip_value", type=float, default=0.1, help="lower and upper clip value for disc. weights")
    parser.add_argument("--device", type=int, default=0, help="the ID of the gpu to run on")
    opt = parser.parse_args()
    print(opt)

    os.makedirs("../images", exist_ok=True)
    out_dir = 'runs/{}'.format(datetime.datetime.now())
    writer = SummaryWriter(out_dir)

    if torch.cuda.is_available():
        cuda = True
        print("Using CUDA")
    else:
        cuda = False
        print("NOT using CUDA")
    device = torch.device("cuda:{}".format(opt.device) if cuda else "cpu")

    # Loss functions
    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    generator = ESRGAN(1, 1)
    # generator = Generator()
    discriminator = Discriminator()

    # Initialize xavier
    generator.apply(init_weights)
    discriminator.apply(init_weights)

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    training_params = {"batch_size": opt.batch_size, "shuffle": True, "num_workers": 0}
    # training_data = ConditionalDataset(device=device)
    training_data = Dataset(device=device)
    train_idx, test_idx = training_data.train_test_split_ids(how='seq')
    training_data.select_indices(train_idx, shuffle=True)  # TODO: this might cause problems!
    training_generator = torch.utils.data.DataLoader(training_data, **training_params)

    validation_params = {"batch_size": opt.batch_size, "shuffle": False, "num_workers": 0}
    validation_data = Dataset(device=device)
    validation_data.select_indices(test_idx, shuffle=False)  # here order doesn't matter
    validation_generator = torch.utils.data.DataLoader(validation_data, **validation_params)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    # ----------
    #  Training
    # ----------
    iter_d = 0
    iter_g = 0
    for epoch in range(opt.n_epochs):
        print("Starting epoch %d" % epoch)

        for i, (pred_imgs, real_imgs) in enumerate(training_generator):

            batch_size = pred_imgs.shape[0]
            if torch.any(pred_imgs.isnan()):
                warnings.warn("Skipping batch with nan value")
                continue

            # Configure input
            real_imgs = Variable(real_imgs.type(FloatTensor))
            pred_imgs = Variable(pred_imgs.type(FloatTensor))

            # ---------------------
            #  Train Discriminator
            # ---------------------
            for p in discriminator.parameters():  # reset requires_grad
                p.requires_grad = True

            optimizer_D.zero_grad()

            # Sample noise as generator input
            noise = Variable(torch.tensor(np.random.normal(0, 1, (batch_size, 8, 32, 48))).type(FloatTensor))

            # Generate a batch of images
            gen_imgs = generator(pred_imgs)

            # Adversarial loss
            # grad_penalty = calc_gradient_penalty(discriminator, real_imgs.data, gen_imgs.data, pred_imgs, batch_size, cuda, opt.device, 10)
            d_loss = -torch.mean(discriminator(real_imgs, pred_imgs)) + torch.mean(discriminator(gen_imgs, pred_imgs))

            d_loss.backward()
            # grad_penalty.backward()
            optimizer_D.step()

            # Clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-opt.clip_value, opt.clip_value)

                # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            iter_d += 1
            writer.add_scalar('Discriminator loss', d_loss.item(), iter_d)

            # Train the generator every n_critic iterations
            if i % opt.n_critic == 0:
                # -----------------
                #  Train Generator
                # -----------------
                for p in discriminator.parameters():
                    p.requires_grad = False
                optimizer_G.zero_grad()

                # Generate a batch of images
                gen_imgs = generator(pred_imgs)
                # Adversarial loss
                g_loss = -torch.mean(discriminator(gen_imgs, pred_imgs))

                g_loss.backward()
                optimizer_G.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, i, len(training_generator), d_loss.item(), g_loss.item())
                )

                iter_g += 1
                writer.add_scalar('Generator loss', g_loss.item(), iter_g)

            batches_done = epoch * len(training_generator) + i
            if batches_done % opt.sample_interval == 0:
                # sample_image(validation_data, n_row=4, batches_done=batches_done, generator=generator, device=device)
                # # writer.add_scalar('CRPS', crps, batches_done)
                # writer.add_scalar('Log spectral distance', lsd, batches_done)
                gen_imgs = torch.squeeze(gen_imgs, 1).detach().numpy()
                real_imgs = torch.squeeze(real_imgs, 1).detach().numpy()
                for i in range(gen_imgs.shape[0]):
                    print(log_spectral_distance_pairs_avg(gen_imgs, real_imgs))
                    plot_power_spectrum(power_spectrum_dB(gen_imgs[i,:,:]), "Power spectrum")
                # writer.add_figure('Generated images', plot_images(gen_imgs, real_imgs, pred_imgs, batches_done),
                #                   global_step=batches_done)
                writer.add_figure('Power spectrum generated', plot_power_spectrum(power_spectrum_batch_avg(gen_imgs), "Power spectrum generated"),
                                  global_step=batches_done)

        torch.save(generator.state_dict(), '%s/generator_epoch_%d.pth' % (out_dir, epoch))
        torch.save(discriminator.state_dict(), '%s/discriminator_epoch_%d.pth' % (out_dir, epoch))
