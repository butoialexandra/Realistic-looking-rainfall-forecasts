
import torch

import os
import random
import warnings
import time

from os.path import join
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from tqdm.auto import tqdm
import numpy as np
import xarray as xr
import pyproj

from sklearn.model_selection import train_test_split

class Dataset(torch.utils.data.Dataset):
    def __init__(self, device='cpu'):
        self.device = device
        self.on_cluster = False
        self.selected_indices = []
        self.observations = self.load_observations()
        self.cosmo = self.load_predictions()
        self.compute_nearest_neighbors()
        self.base_dir = "/mnt/ds3lab-scratch/bhendj/data"


    def load_observations(self, verbose=True):
        """
        Observations: data for every hour
            since 2016-05
            until 2020-09
            x y: 710 x 640
        """
        files = {}
        if self.on_cluster:
            dt = datetime(year=2016, month=5, day=1)
            while dt < datetime(year=2020, month=10, day=1):
                path = join(self.base_dir, "combiprecip", f"combiprecip_{dt.year}{dt.month:02d}.nc")
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

    def load_predictions(self):
        """
        Predictions: based on 00:00 and 12:00 (reftime)
            data for every hour in next 5 days (leadtime is timedelta)
            x y: 188 x 127
        """
        if self.on_cluster:
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

    def __len__(self):
        return len(self.selected_indices)

    def get_obs_by_id(self, id):
        key, id, reftime, leadtime = id.split("-", maxsplit=3)
        id = int(id)
        return self.observations[key].isel(time=id, dummy=0)

    def compute_nearest_neighbors(self):

        chx, chy = self.cosmo['chx'], self.cosmo['chy']
        for key, obs in self.observations.items():
            self.observations[key] =  obs.sel(chx=chx, chy=chy, method='nearest')

    def get_x_y_by_id(self, id: str):
        # TODO: list all times
        t = time.time()
        obs_point = self.get_obs_by_id(id)
        target_time = obs_point['time'].values
        target_time_py = datetime.utcfromtimestamp(target_time.tolist() / 1e9)
        hours_from_midnight = np.timedelta64(target_time_py.hour, 'h')

        # Get latest valid prediction
        reftime = self.cosmo['reftime'].where(
            self.cosmo['reftime'] < target_time,
            drop=True
        )[-1].values
        leadtime = int((target_time - reftime) / np.timedelta64(1, 'h'))

        if leadtime <= 0:
            raise Exception(f"Invalid leadtime {leadtime} for target_time {target_time} and reftime {reftime}")


        pred_points = self.cosmo.sel(reftime=reftime).isel(leadtime=leadtime, member=0)  # TODO: member!
        real_point = obs_point
        prec_pred = pred_points['PREC'].values
        prec_real = real_point['RR'].values
        if np.any(np.isnan(prec_pred)):
            warnings.warn(f"nan value encountered in ensemble for reftime {reftime} leadtime {leadtime}")

        return prec_pred, prec_real  # X: predictions, Y: observations

    def get_image_ids(self):
        image_ids = []
        for key, obs in self.observations.items():
            assert len(obs['time'].values) < 2000
            val = [f"{key}-{i}-{dt}" for i, dt in enumerate(obs['time'].values)]
            image_ids += val
        return image_ids

    def train_test_split_ids(self, how='random', test_size=0.1):
        ids = self.get_image_ids()
        if how == 'random':
            train_ids, test_ids = train_test_split(ids, test_size=test_size)
            return train_ids, test_ids
        elif how == 'seq':
            split_idx = int(len(ids) * (1 - test_size))
            train_ids = ids[:split_idx]
            test_ids = ids[split_idx:]
            return train_ids, test_ids
            # TODO: another 'how': split by month or something
        else:
            raise Exception("Invalid argument")

    def select_indices(self, indices, shuffle=True):
        self.selected_indices = indices.copy()
        if shuffle:
            random.shuffle(self.selected_indices)
        return self

    def select_all(self):
        self.select_indices(self.get_image_ids(), shuffle=False)
        return self

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

    ax = fig.add_subplot(122)
    ax.set_title('Observation')
    plt.imshow(prec_real)

    plt.colorbar(orientation='horizontal')
    plt.savefig("foo.png")


def save_buffer_to_file(buffer, filename):
    buffer = torch.cat(buffer, dim=0)
    torch.save(buffer, filename)


def create_training_dataset(out_dir="/mnt/ds3lab-scratch/mzilinec/preprocessed/"):
    ds = Dataset(device='cpu')
    ds.select_all()
    assert len(ds) > 0
    buffer_x, buffer_y = [], []
    file_offset = 0

    for i, (x, y) in tqdm(enumerate(ds), total=len(ds)):

        if torch.any(x.isnan()) or torch.any(y.isnan()):
            continue

        buffer_x.append(x)
        buffer_y.append(y)

        if len(buffer_x) > 1024:
            save_buffer_to_file(buffer_x, join(out_dir, f"x.{file_offset}.pt"))
            save_buffer_to_file(buffer_y, join(out_dir, f"y.{file_offset}.pt"))
            file_offset += 1
            buffer_x, buffer_y = [], []

    if len(buffer_x) > 0:
        save_buffer_to_file(buffer_x, join(out_dir, f"x.{file_offset}.pt"))
        save_buffer_to_file(buffer_y, join(out_dir, f"y.{file_offset}.pt"))


if __name__ == "__main__":
    create_training_dataset()
    # d = Dataset()
    # prec_pred, prec_real = d.get_x_y_at_time(0)
    # plotit()
