import argparse
import os
import datetime
import warnings

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

from dataset import Dataset
from modules import Generator, Discriminator
from util import init_weights, sample_image, plot_image

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling")
    opt = parser.parse_args()
    print(opt)

    os.makedirs("images", exist_ok=True)
    writer = SummaryWriter('runs/{}'.format(datetime.datetime.now()))

    # input_shape = (127, 188)
    # output_shape = (295, 427)
    # TODO: make sure we didn't mix up x, y from dataset

    if torch.cuda.is_available():
        cuda = True
        print("Using CUDA")
    else:
        cuda = False
        print("NOT using CUDA")
    device = torch.device("cuda:0" if cuda else "cpu")

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

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    # ----------
    #  Training
    # ----------
    iter = 0
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
            d_real_loss = adversarial_loss(validity_real, valid)

            # Loss for fake images
            fake_imgs = gen_imgs.detach()
            validity_fake = discriminator(fake_imgs, pred_imgs)
            d_fake_loss = adversarial_loss(validity_fake, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(training_generator), d_loss.item(), g_loss.item())
            )

            iter += 1
            writer.add_scalar('Discriminator loss', d_loss.item(), iter)
            writer.add_scalar('Generator loss', g_loss.item(), iter)

            batches_done = epoch * len(training_generator) + i
            if batches_done % opt.sample_interval == 0:
                lsd = sample_image(training_data, n_row=10, batches_done=batches_done, generator=generator, device=device)
                # writer.add_scalar('CRPS', crps, batches_done)
                writer.add_scalar('Log spectral distance', lsd, batches_done)
                writer.add_figure('Generated images', plot_image(training_data, generator, device),
                                  global_step=batches_done)
