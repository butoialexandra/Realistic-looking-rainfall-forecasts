
import torch

import os
import random
import warnings
import glob
import time

from os.path import join
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from tqdm.auto import tqdm
import numpy as np
import xarray as xr
import pyproj


from sklearn.model_selection import train_test_split

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

def load_predictions_from_cache(load_dir, verbose=True):
    predictions = []
    for filename in glob.glob(f"{load_dir}/predictions*.pt"):
        predictions.append(torch.load(filename).numpy())
        print(f"Loaded {filename}")
    predictions = np.concatenate(predictions,axis=0) 
    return predictions

class UnconditionalDataset(torch.utils.data.Dataset):
    def __init__(self, device='cpu', on_cluster=False):
        self.device = device
        self.base_dir = "/mnt/ds3lab-scratch/bhendj/data"
        load_dir = "/mnt/ds3lab-scratch/dslab2019/shghosh/preprocessed"
        if os.path.isdir(load_dir):
            self.predicted_images = load_predictions_from_cache(load_dir)
            #self.predicted_images = self.predicted_images[:, :, 30:158]
            zero_mat1 = (np.random.rand(self.predicted_images.shape[0],1,self.predicted_images.shape[2]) + 0.5).astype(float)
            self.predicted_images = np.concatenate((self.predicted_images, zero_mat1), axis=1)
            zero_mat2 = (np.random.rand(self.predicted_images.shape[0],self.predicted_images.shape[1],4) + 0.5).astype(float)
            self.predicted_images = np.concatenate((self.predicted_images, zero_mat2), axis=2)
            median_val = np.median(self.predicted_images.sum(axis=(1,2))) + 100.0
            self.predicted_images = self.predicted_images[self.predicted_images.sum(axis=(1,2))> median_val]
        else:
            self.cosmo = load_predictions(self.base_dir, on_cluster)
            self.load_images_and_remove_nan()
        self.standardize_images()

    def load_images_and_remove_nan(self):
        self.predicted_images = []
        for leadtime in range(1, self.cosmo.leadtime.values.shape[0]):
            self.predicted_images.append(self.cosmo.PREC.isel(leadtime=leadtime, member=0).values) #TODO: Member
        self.predicted_images = np.array(self.predicted_images)
        self.predicted_images = self.predicted_images[:, 32:96, 62:126]
        #Removing NaN indices
        nan_indices = np.unique(np.isnan(self.predicted_images).nonzero()[0])
        self.predicted_images = np.delete(self.predicted_images, nan_indices, axis=0)

    def standardize_images(self):
        self.normalize()
        #Standardizing to 0.5 mean, 0.5 std
        # self.predicted_images -= np.mean(self.predicted_images, axis=0, keepdims=True)
        # self.predicted_images /= 2.0*np.std(self.predicted_images, axis = 0, keepdims=True)
        # self.predicted_images += 0.5
        # self.predicted_images = np.expand_dims(self.predicted_images, axis=1)
    
    def normalize(self):
        # Mean              Stddev              Min      Max
        # 0.314654152002613 1.0515969624985164 -0.015625 97.326171875
        self.predicted_images /= 10.0
        # 0.03147736358815389 0.10518044480120241 -0.0015625 9.7326171875
        self.predicted_images = np.clip(self.predicted_images, a_min=0.0, a_max=1.0)
        # print(self.predicted_images.mean(), self.predicted_images.std(), self.predicted_images.min(), self.predicted_images.max())
        # raise
        self.predicted_images = np.expand_dims(self.predicted_images, axis=1)

    def __len__(self):
        return len(self.predicted_images)

    def __getitem__(self, index):
        img = self.predicted_images[index]
        img = torch.tensor(img, device=self.device)
        return img

class Dataset(torch.utils.data.Dataset):
    def __init__(self, device='cpu', on_cluster=False):
        self.device = device
        self.selected_indices = []
        self.base_dir = "/mnt/ds3lab-scratch/bhendj/data"
        self.observations = load_observations(self.base_dir, on_cluster)
        self.cosmo = load_predictions(self.base_dir, on_cluster)
        self.compute_nearest_neighbors()



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
