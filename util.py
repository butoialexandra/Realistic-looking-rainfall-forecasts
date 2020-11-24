import matplotlib.pyplot as plt
import torch
from torch.nn import init

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
    y_pred = torch.tensor(y_pred, device=device).repeat(n_row, 1, 1) / 10
    y_real = torch.tensor(y_real, device=device).repeat(n_row, 1, 1) / 10
    y_pred = y_pred.unsqueeze(1)
    y_real = y_real.unsqueeze(1)
    gen_imgs = generator(y_pred)

    fig = plt.figure(figsize=(6, 3.2))
    ax = fig.add_subplot(121)
    ax.set_title('Output')
    plt.imshow(gen_imgs.squeeze().detach().cpu())

    ax = fig.add_subplot(122)
    ax.set_title('Observation')
    plt.imshow(y_real.squeeze().detach().cpu())
    plt.colorbar(orientation='horizontal')
    plt.savefig("images/%d.png" % batches_done)