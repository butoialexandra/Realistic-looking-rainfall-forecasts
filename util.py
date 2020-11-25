import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
from torch.nn import init
from verification import crps, log_spectral_distance


def reproject_to_swiss_coords(cosmo):
    """
    Reproject the COSMO coordinates to Swiss coordinates

    :param cosmo: cosmo xarray dataset
    :return: reprojected dataset
    """
    src_proj = pyproj.Proj("EPSG:4326")
    dst_proj = pyproj.Proj("EPSG:21781")

    src_x = cosmo.lon.values
    src_y = cosmo.lat.values

    dst_x, dst_y = pyproj.transform(src_proj, dst_proj, src_x, src_y, always_xy=True)
    cosmo = cosmo.assign_coords({"chx": (("y", "x"), dst_x), "chy": (("y", "x"), dst_y)})

    return cosmo


def get_nn(cosmo_img, combi_img):
    """
    Given a cosmo image, find nearest points in the observation image
    :param cosmo_img:
    :param combi_img:
    :return:
    """
    chx, chy = cosmo_img['chx'], cosmo_img['chy']
    real_points = combi_img.sel(chx=chx, chy=chy, method='nearest')

    return real_points


if __name__ == '__main__':
    cosmo = xr.open_zarr("cosmoe_prec_201805.zarr")
    cosmo = reproject_to_swiss_coords(cosmo)
    ds = xr.open_mfdataset("combiprecip_201805.nc", combine='by_coords')

    forecast = cosmo.isel(reftime=0, leadtime=1)
    observations = ds.isel(time=0, dummy=0)

    start = time.time()
    real_points = get_nn(forecast, observations)
    end = time.time()
    print("Time elapsed: {}".format(end - start))

    fig, axes = plt.subplots(ncols=2, figsize=(18, 8))
    forecast.PREC.isel(member=0).plot.pcolormesh("chx", "chy", ax=axes[0], cmap='viridis', vmin=0)
    real_points.RR.plot.pcolormesh("chx", "chy", ax=axes[1])
    plt.show()


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


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
    gen_imgs = generator(y_pred)

    fig = plt.figure(figsize=(6, 3.2))
    ax = fig.add_subplot(121)
    ax.set_title('Output')
    plt.imshow(gen_imgs.squeeze().detach().cpu())

    ax = fig.add_subplot(122)
    ax.set_title('Observation')
    plt.imshow(y_real.squeeze().detach().cpu())

    return fig
