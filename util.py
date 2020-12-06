import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.nn import init
from torch.autograd import Variable
from verification import crps, log_spectral_distance

def init_weights(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def sample_image(training_data, n_row, batches_done, generator, device):
    n_row = 1 # FIXME
    # Sample noise
    # z = Variable(FloatTensor(np.random.normal(0, 1, (n_row, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    y_pred, y_real = training_data.get_x_y_by_id(training_data.selected_indices[0])  # TODO: fix first date from 201805
    y_pred = torch.tensor(y_pred, device=device).repeat(n_row, 1, 1)
    y_real = torch.tensor(y_real, device=device).repeat(n_row, 1, 1)
    y_pred = y_pred.unsqueeze(1)
    y_real = y_real.unsqueeze(1)
    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    noise = Variable(torch.tensor(np.random.normal(0, 1, (1, 8, 32, 32))).type(FloatTensor))
    gen_imgs = generator(y_pred, noise)

    fig = plt.figure(figsize=(6, 3.2))
    ax = fig.add_subplot(121)
    ax.set_title('Output')
    plt.imshow(gen_imgs.squeeze().detach().cpu())

    ax = fig.add_subplot(122)
    ax.set_title('Observation')
    plt.imshow(y_real.squeeze().detach().cpu())
    plt.colorbar(orientation='horizontal')
    plt.savefig("images/%d.png" % batches_done)

    gen_imgs = gen_imgs.squeeze(1).squeeze(0).detach().cpu().numpy()
    y_real = y_real.squeeze(1).squeeze(0).detach().cpu().numpy()
    # crps_score = crps(gen_imgs, y_real)
    lsd_score = log_spectral_distance(gen_imgs, y_real)
    return lsd_score

def plot_image(training_data, generator, device):
    n_row = 1  # FIXME
    y_pred, y_real = training_data.get_x_y_by_id(training_data.selected_indices[0])  # TODO: fix first date from 201805
    y_pred = torch.tensor(y_pred, device=device).repeat(n_row, 1, 1) / 10
    y_real = torch.tensor(y_real, device=device).repeat(n_row, 1, 1) / 10
    y_pred = y_pred.unsqueeze(1)
    y_real = y_real.unsqueeze(1)
    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    noise = Variable(torch.tensor(np.random.normal(0, 1, (1, 8, 32, 32))).type(FloatTensor))
    gen_imgs = generator(y_pred, noise)

    fig = plt.figure(figsize=(6, 3.2))
    ax = fig.add_subplot(121)
    ax.set_title('Output')
    plt.imshow(gen_imgs.squeeze().detach().cpu())
    plt.colorbar()

    ax = fig.add_subplot(122)
    ax.set_title('Observation')
    plt.imshow(y_real.squeeze().detach().cpu())
    plt.colorbar()

    return fig