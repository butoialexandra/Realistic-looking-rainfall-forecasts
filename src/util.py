import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.nn import init
from torch.autograd import Variable
from verification import crps, log_spectral_distance
import random

def init_weights(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

# def sample_image(data, n_row, batches_done, generator, device):
#     # generate small batch of images
#     shuffled_indices = random.sample(data.selected_indices, 4)
#     y_pred, y_real = data.get_x_y_by_id(shuffled_indices[0])
#     y_pred, y_real = pad(y_pred, y_real)
#     y_pred = torch.tensor(y_pred, device=device).repeat(1, 1, 1).unsqueeze(1)
#     y_real = torch.tensor(y_real, device=device).repeat(1, 1, 1).unsqueeze(1)
#     batch_size = min(n_row, len(shuffled_indices))
#     for i in range(1, batch_size):
#         p, r = data.get_x_y_by_id(shuffled_indices[i])
#         p, r = pad(p, r)
#         y_pred = torch.cat([
#             torch.tensor(p, device=device).repeat(1, 1, 1).unsqueeze(1),
#             y_pred
#         ], 0)
#         y_real = torch.cat([
#             torch.tensor(r, device=device).repeat(1, 1, 1).unsqueeze(1),
#             y_real
#         ], 0)
#
#     # generate images
#     FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
#     noise = Variable(torch.tensor(np.random.normal(0, 1, (n_row, 8, 32, 48))).type(FloatTensor))
#     y_pred, y_real = Variable(y_pred.type(FloatTensor)), Variable(y_real.type(FloatTensor))
#     gen_imgs = generator(y_pred, noise)
#
#     # plot figure
#     fig = plt.figure(figsize=(9, 3.2*n_row))
#     for i in range(batch_size):
#         ax = fig.add_subplot(n_row, 3, i * 3 + 1)
#         ax.set_title('Forecast')
#         plt.imshow(y_pred[i, :, :, :].squeeze().detach().cpu())
#         plt.colorbar(orientation='vertical')
#
#         ax = fig.add_subplot(n_row, 3, i*3+2)
#         ax.set_title('Observation')
#         plt.imshow(y_real[i, :, :, :].squeeze().detach().cpu())
#         plt.colorbar(orientation='vertical')
#
#         ax = fig.add_subplot(n_row, 3, i*3+3)
#         ax.set_title('Output')
#         plt.imshow(gen_imgs[i, :, :, :].squeeze().detach().cpu())
#         plt.colorbar(orientation='vertical')
#     plt.savefig("images/%d.png" % batches_done)

def plot_image(data, n_row, batches_done, generator, device):
    # generate small batch of images
    # shuffled_indices = random.sample(data.selected_indices, 4)
    shuffled_indices = data.selected_indices[:4]
    y_pred, y_real = data.get_x_y_by_id(shuffled_indices[0])
    y_pred, y_real = pad(y_pred, y_real)
    y_pred = torch.tensor(y_pred, device=device).repeat(1, 1, 1).unsqueeze(1)
    y_real = torch.tensor(y_real, device=device).repeat(1, 1, 1).unsqueeze(1)
    batch_size = min(n_row, len(shuffled_indices))
    for i in range(1, batch_size):
        p, r = data.get_x_y_by_id(shuffled_indices[i])
        p, r = pad(p,r)
        y_pred = torch.cat([
            torch.tensor(p, device=device).repeat(1, 1, 1).unsqueeze(1),
            y_pred
        ], 0)
        y_real = torch.cat([
            torch.tensor(r, device=device).repeat(1, 1, 1).unsqueeze(1),
            y_real
        ], 0)

    # generate images
    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    noise = Variable(torch.tensor(np.random.normal(0, 1, (n_row, 8, 32, 48))).type(FloatTensor))
    y_pred, y_real = Variable(y_pred.type(FloatTensor)), Variable(y_real.type(FloatTensor))
    gen_imgs = generator(y_pred, noise)

    # plot figure
    fig = plt.figure(figsize=(9, 3.2 * n_row))
    for i in range(batch_size):
        ax = fig.add_subplot(n_row, 3, i * 3 + 1)
        ax.set_title('Forecast')
        plt.imshow(y_pred[i, :, :, :].squeeze().detach().cpu())
        plt.colorbar(orientation='vertical')

        ax = fig.add_subplot(n_row, 3, i * 3 + 2)
        ax.set_title('Observation')
        plt.imshow(y_real[i, :, :, :].squeeze().detach().cpu())
        plt.colorbar(orientation='vertical')

        ax = fig.add_subplot(n_row, 3, i * 3 + 3)
        ax.set_title('Output')
        plt.imshow(gen_imgs[i, :, :, :].squeeze().detach().cpu())
        plt.colorbar(orientation='vertical')
    plt.savefig("images/%d.png" % batches_done)

    return fig


def pad(x, y):
    pred = np.zeros((128, 192))
    pred[:-1, :-4] = x
    obs = y[:256, :384]
    pred = np.clip(pred, 0, 20) / 20
    obs = np.clip(obs, 0, 20) / 20

    return pred, obs