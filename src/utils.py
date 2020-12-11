import torch

import numpy as np
import xarray as xr
import pyproj
import torch

import glob
from os.path import join
import matplotlib.pyplot as plt

#Load observations

def load_observations(base_dir, on_cluster=False, verbose=True):
    """
    Observations: data for every hour
        since 2016-05
        until 2020-09
        x y: 710 x 640
    """
    files = {}
    if on_cluster:
        dt = datetime(year=2016, month=5, day=1)
        while dt < datetime(year=2020, month=10, day=1):
            path = join(base_dir, "combiprecip", f"combiprecip_{dt.year}{dt.month:02d}.nc")
            if verbose:
                print(path)
            obj = xr.open_mfdataset(path, combine='by_coords')
            files[f"{dt.year}{dt.month:02d}"] = obj
            dt = dt + relativedelta(months=1)
    else:
        path = "./combiprecip_201805.nc"
        weather = xr.open_mfdataset(path, combine='by_coords')
        files['201805'] = weather
    return files


#Load Predictions

def load_predictions(base_dir, on_cluster=False):
    """
    Predictions: based on 00:00 and 12:00 (reftime)
        data for every hour in next 5 days (leadtime is timedelta)
        x y: 188 x 127
    """
    if on_cluster:
        path = join(base_dir, "cosmoe", "data.zarr", "data_ethz.zarr")
    else:
        path = "./cosmoe_prec_201805.zarr"

    cosmo = xr.open_zarr(path)

    # Transform to the other coordinate system
    src_proj = pyproj.Proj("EPSG:4326") # WSG84
    dst_proj = pyproj.Proj("EPSG:21781") # CH1903 / LV03
    src_x = cosmo.lon.values
    src_y = cosmo.lat.values
    dst_x, dst_y = pyproj.transform(src_proj, dst_proj, src_x, src_y, always_xy=True)
    cosmo = cosmo.assign_coords({"chx": (("y", "x"), dst_x) , "chy": (("y", "x"), dst_y)})
    return cosmo


# Load Observations from Cache

def load_obs_from_cache(load_dir, highres = True, verbose=True):
    """
    Cached Observations: data for every hour
        since 2016-05
        until 2020-09
        x y: 710 x 640
    """
    observations = []
    if highres:
        file_idx = "observations_hr"
    else:
        file_idx = "observations"
    for filename in glob.glob(f"{load_dir}/{file_idx}*.pt"):
        observations.append(torch.load(filename).numpy())
        if verbose:
            print(f"Loaded {filename}")
    observations = np.concatenate(observations, axis=0)
    return observations


# Load predictions from cache

def load_predictions_from_cache(load_dir, verbose=True):
    """
    Cached Predictions: based on 00:00 and 12:00 (reftime)
        data for every hour in next 5 days (leadtime is timedelta)
        x y: 188 x 127
    """
    predictions = []
    for filename in glob.glob(f"{load_dir}/predictions*.pt"):
        predictions.append(torch.load(filename).numpy())
        if verbose:
            print(f"Loaded {filename}")
    predictions = predictions.concatenate(predictions, axis=0)
    return predictions


#Plot images

def plot_images_grid(images, save_path):
    ndata = images.shape[0]
    ncols = 5
    nrows = int(ndata/ncols) + 1
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 5*nrows))
    for i in range(ndata):
        x =  i/ncols
        y = i%ncols
        im = axs[x,y].pcolormesh(images[i], vmin=0, vmax=1, cmap='viridis')
        fig.colorbar(im, ax=axs[x,y])
    plt.savefig(save_path)
    return fig

#Plot function to compare generated image and real image for conditional GAN

def plot_images_compare(gen_img, real_img, save_path):
    fig,axs = plt.subplots(nrows=1, ncols=2, figsize =(15,7))
    im = axs[0].pcolormesh(gen_img, vmin=0, vmax=1, cmap='viridis')
    axs[0].set_title('Generated image')
    fig.colorbar(im, ax=axs[0])
    im = axs[1].pcolormesh(real_img, vmin=0, vmax=1, cmap='viridis')
    axs[1].set_title('Real image')
    fig.colorbar(im, ax=axs[1])
    plt.savefig(save_path)

#Plot single image
def plot_image(image, save_path):
    plt.title('Unconditional Generated Image')
    plt.pcolormesh(image, vmin=0, vmax=1, cmap='viridis')
    plt.savefig(save_path)

#Tiling images
def create_tiles(predictions, observations, highres, img_size):
    """

    """
    prediction_tiles = []
    observation_tiles = []

    start_idx = 0


#Binning images
def create_bins(images):
    """

    """
    pass


#Create Train Test split
def train_test_split(images):
    pass

#Function for standardizing images
def standardize_images_cliiping(images):
    """
    Scales values and clips images to lie between 0 and 1
    """
    images /= 10.0
    images = np.clp(images, a_min=0.0, a_max=0.0)
    return images

# Weight initialization
def weights_init(m):

    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

##################################################################################################################################

# function to plot images
def plot_images(images, path):
    fig, axs =  plt.subplots(ncols=8, nrows=8, figsize=(16,16))
    #nrow = np.sqrt(images.shape[0])
    for i in range(images.shape[0]):
        row_num = int(i / 8)
        col_num = int(i % 8)
        im = axs[row_num, col_num].pcolormesh(images[i,0], vmin=0, vmax=1, cmap='viridis')
        fig.colorbar(im, ax=axs[row_num, col_num])
    plt.savefig(path)


#Function to plot single image pair -- conditional
def plot_image_single_conditional(generated, observed, path):
    plt.figure(figsize=(6,3.2))
    fig, ax  = plt.subplots(1,2)
    ax[0].set_title("Generated")
    ax[0].imshow(generated, vmin=0, vmax=1)
    ax[1].set_title("Observed")
    ax[1].imshow(observed, vmin=0, vmax=1)
    plt.savefig(path)

#Function to plot single image unconditional
def plot_image_single_unconditional(generated, path):
    plt.figure(figsize=(4,4))
    plt.imshow(generated, vmin=0, vmax=1)
    plt.savefig(path)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


def upscale_input(x, mode='copy'):
    if mode == 'copy':
        # This copies every element in the 2nd and 3rd dims twice.
        # Example in 1D: [1,2,3] -> [1,1,2,2,3,3]
        x = torch.repeat_interleave(x, 2, dim=2)
        x = torch.repeat_interleave(x, 2, dim=3)
        return x
    elif mode == 'interpolate':
        raise Exception("Not implemented")
    raise Exception(f"Unknown upscale mode: {mode}")


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
