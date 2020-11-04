
import torch

import os
import random
from os.path import join
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
import xarray as xr
import pyproj

from sklearn.model_selection import train_test_split


base_dir = "/mnt/ds3lab-scratch/bhendj/data"
on_cluster = True

def load_observations(verbose=True):
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


def load_predictions():
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


class Dataset(torch.utils.data.Dataset):
    def __init__(self, device='cpu'):
        self.device = device
        self.selected_indices = []
        self.observations = load_observations()
        self.cosmo = load_predictions()
        self.top_left, self.bottom_right = self.compute_nearest_neighbors()
    
    def __len__(self):
        return len(self.selected_indices)

    def get_obs_by_id(self, id):
        key, id, reftime, leadtime = id.split("-", maxsplit=3)
        id = int(id)
        return self.observations[key].isel(time=id, dummy=0)

    def compute_nearest_neighbors(self):
        real_point = self.observations['201805']#.isel(reftime=0, leadtime=0, member=0)
        pred_point = self.cosmo
        chx, chy = pred_point['chx'].min(), pred_point['chy'].min()
        top_left = real_point.sel(chx=chx, chy=chy, method='nearest')
        # TODO: is it ok or should it be always nearest "inside"?
        chx, chy = pred_point['chx'].max(), pred_point['chy'].max()
        bottom_right = real_point.sel(chx=chx, chy=chy, method='nearest')
        return (top_left['chx'].values, top_left['chy'].values), (bottom_right['chx'].values, bottom_right['chy'].values)

    def get_x_y_by_id(self, id: str):
        # TODO: list all times
        obs_point = self.get_obs_by_id(id)
        target_time = obs_point['time'].values
        target_time_py = datetime.utcfromtimestamp(target_time.tolist() / 1e9)
        leadtime = np.timedelta64(target_time_py.hour, 'h')
        reftime = target_time - leadtime
        leadtime = int((leadtime / np.timedelta64(1, 'h'))) - 1

        pred_points = self.cosmo.sel(reftime=reftime).isel(leadtime=leadtime, member=0)  # TODO: member!

        # Crop real image to the boundaries of predicted image
        real_point = obs_point.where(
            (obs_point['chx'] >= self.top_left[0]) & (obs_point['chx'] <= self.bottom_right[0])
          & (obs_point['chy'] >= self.top_left[1]) & (obs_point['chy'] <= self.bottom_right[1]) 
        , drop=True)  
        # The cropped reference should be [640 x 710] -> [295 x 427] vs. input [127 x 188]
        # In other words, the input is 23,876 points, and the reference is 125,965 points

        prec_pred = pred_points['PREC'].values
        prec_real = real_point['RR'].values
        print("Input:", prec_pred.shape, "Reference:", prec_real.shape)
        return prec_pred, prec_real  # X: predictions, Y: observations

    def get_image_ids(self):
        image_ids = []
        for key, obs in self.observations.items():
            assert len(obs['time'].values) < 2000
            val = [f"{key}-{i}-{dt}" for i, dt in enumerate(obs['time'].values)]
            image_ids += val
        return image_ids
    
    def train_test_split_ids(self, how='random', test_size=0.1):
        if how == 'random':
            ids = self.get_image_ids()
            train_ids, test_ids = train_test_split(ids, test_size=test_size)
            return train_ids, test_ids
        else:
            raise Exception("Not implemented")

    def select_indices(self, indices, shuffle=True):
        self.selected_indices = indices.copy()
        random.shuffle(self.selected_indices)

    def __getitem__(self, index):
        x, y = self.get_x_y_by_id(self.selected_indices[index])
        x = torch.tensor(x, device=self.device)
        y = torch.tensor(y, device=self.device)
        return x, y


def plotit():
    import numpy as np
    import matplotlib.pyplot as plt
    
    prec_pred, prec_real = d.get_x_y_at_time(0)

    fig = plt.figure(figsize=(6, 3.2))

    ax = fig.add_subplot(121)
    ax.set_title('Prediction #0')
    plt.imshow(prec_pred)
    # ax.set_aspect('equal')

    ax = fig.add_subplot(122)
    ax.set_title('Observation')
    plt.imshow(prec_real)
    # ax.set_aspect('equal')

    # cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    # cax.get_xaxis().set_visible(False)
    # cax.get_yaxis().set_visible(False)
    # cax.patch.set_alpha(0)
    # cax.set_frame_on(False)
    plt.colorbar(orientation='horizontal')
    plt.savefig("foo.png")


if __name__ == "__main__":
    d = Dataset()
    prec_pred, prec_real = d.get_x_y_at_time(0)
    plotit()
