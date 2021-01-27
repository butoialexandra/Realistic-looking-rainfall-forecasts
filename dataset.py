
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
        if verbose:
            print(f"Loaded {filename}")
    predictions = np.concatenate(predictions,axis=0) 
    return predictions

def load_x_y_from_cache(load_dir, test=True,verbose=True):
    predictions = []
    observations = []
    observations_hr = []
    if test:
        idx_set = [1,2]
    else:
        idx_set = range(1,37)
    for idx in idx_set:
        filename = f"{load_dir}/predictions.{idx}.pt"
        predictions.append(torch.load(filename).numpy())
        if verbose:
            print(f"Loaded {filename}")
    for idx in idx_set:
        filename = f"{load_dir}/observations.{idx}.pt"
        observations.append(torch.load(filename).numpy())
        if verbose:
            print(f"Loaded {filename}")
    for idx in idx_set:
        filename = f"{load_dir}/observations_hr.{idx}.pt"
        observations_hr.append(torch.load(filename).numpy())
        if verbose:
            print(f"Loaded {filename}")
    predictions = np.concatenate(predictions,axis=0)
    observations = np.concatenate(observations, axis=0)
    observations_hr = np.concatenate(observations_hr, axis=0)
    return predictions, observations, observations_hr


class UnconditionalDatasetObservations(torch.utils.data.Dataset):
    def __init__(self, device='cpu', on_cluster=False, test=False):
        self.device = device
        load_dir = "/mnt/ds3lab-scratch/dslab2019/shghosh/preprocessed"
        if test:
            dataset_len = 2050
        else:
            dataset_len =  37497
        #_,_,self.observed_images = load_x_y_from_cache(load_dir)
        #print("Finished loading images\n")
        prev_time = time.time()
        rand_mat  = (np.zeros((dataset_len, 256, 384))+0.1).astype(float)
        _, _, rand_mat[:, :-3, :-9] = load_x_y_from_cache(load_dir)
        self.observed_images = rand_mat
        #rand_mat = (np.random.rand(self.observed_images.shape[0],3,self.observed_images.shape[2])+0.1).astype(float)
        #self.observed_images = np.concatenate((self.observed_images, rand_mat), axis=1)
        #rand_mat = (np.random.rand(self.observed_images.shape[0],self.observed_images.shape[1],9)+0.1).astype(float)
        #self.observed_images = np.concatenate((self.observed_images, rand_mat), axis=2)
        print("time to append images: ",time.time()-prev_time )
        prev_time = time.time()
        median_val = np.mean(self.observed_images.sum(axis=(1,2)))/2.0
        idx_retain = self.observed_images.sum(axis=(1,2)) > median_val
        self.observed_images = self.observed_images[idx_retain]
        print("time to remove low precip images: ", time.time() - prev_time)
        prev_time = time.time()
        self.standardize_images()
        print("time to standardize images:", time.time() - prev_time)
    
    def __len__(self):
        return len(self.observed_images)

    def standardize_images(self):
        self.observed_images /= 10.0
        self.observed_images = np.clip(self.observed_images, a_min=0.0, a_max=1.0)
        self.observed_images = np.expand_dims(self.observed_images, axis=1)

    def __getitem__(self, index):
        img = self.observed_images[index]
        img = torch.tensor(img, device=self.device)
        return img

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
        #Standardizing to 0.5 mean, 0.5 std
        self.predicted_images -= np.mean(self.predicted_images, axis=0, keepdims=True)
        self.predicted_images /= 2.0*np.std(self.predicted_images, axis = 0, keepdims=True)
        self.predicted_images += 0.5
        self.predicted_images = np.expand_dims(self.predicted_images, axis=1)
    
    def __len__(self):
        return len(self.predicted_images)

    def __getitem__(self, index):
        img = self.predicted_images[index]
        img = torch.tensor(img, device=self.device)
        return img


class ConditionalDataset(torch.utils.data.Dataset):
    def __init__(self, device='cpu', img_size =64, highres= True, test=True):
        load_dir = "/mnt/ds3lab-scratch/dslab2019/shghosh/preprocessed"
        if test:
            dataset_len = 2050
        else:
            dataset_len =  37497
        #self.predictions = (np.zeros((dataset_len, 128, 192))+0.1).astype(float)
        #self.observations = (np.zeros((dataset_len, 128, 192))+0.1).astype(float)
        #self.observations_highres = (np.zeros((dataset_len, 256, 384))+0.1).astype(float)
        #self.predictions[:, :-1, :-4], self.observations[:,:-1,:-4], self.observations_highres[:,:-3, :-9] = load_x_y_from_cache(load_dir)
        self.predictions, self.observations, self.observations_highres = load_x_y_from_cache(load_dir)
        #self.predictions = self.predictions[:len(self.observations)] #Can remove once all the data is cached
        #rand_mat = (np.random.rand(self.predictions.shape[0],1,self.predictions.shape[2])+0.1).astype(float)
        #self.predictions = np.concatenate((self.predictions, rand_mat), axis=1)
        #rand_mat = (np.random.rand(self.predictions.shape[0],self.predictions.shape[1],4)+0.1).astype(float)
        #self.predictions = np.concatenate((self.predictions, rand_mat), axis=2)
        #rand_mat = (np.random.rand(self.observations.shape[0],1,self.observations.shape[2])+0.1).astype(float)
        #self.observations = np.concatenate((self.observations, rand_mat), axis=1)
        #rand_mat = (np.random.rand(self.observations.shape[0],self.observations.shape[1],4)+0.1).astype(float)
        #self.observations = np.concatenate((self.observations, rand_mat), axis=2)
        #rand_mat = (np.random.rand(self.observations_highres.shape[0],3,self.observations_highres.shape[2])+0.1).astype(float)
        #self.observations_highres = np.concatenate((self.observations_highres, rand_mat), axis=1)
        #rand_mat = (np.random.rand(self.observations_highres.shape[0],self.observations_highres.shape[1],9)+0.1).astype(float)
        #self.observations_highres = np.concatenate((self.observations_highres, rand_mat), axis=2)
        # Standardizing images
        self.predictions = self.standardize_images(self.predictions)
        self.observations = self.standardize_images(self.observations)
        self.observations_highres = self.standardize_images(self.observations_highres)
        #median_val = np.mean(self.observations.sum(axis=(1,2,3)))*5.0
        #idx_retain = self.observations.sum(axis=(1,2,3)) > median_val
        sum_precip = (self.observations==1.0).sum(axis=(1,2,3))
        idx_retain = (sum_precip > 0)
        self.predictions = self.predictions[idx_retain]
        self.observations = self.observations[idx_retain]
        self.observations_highres = self.observations_highres[idx_retain]
        self.highres = highres
        self.img_size = img_size
        tile_size_lowres = int(self.img_size/2)
        tile_size_highres = int(self.img_size)
        offset_highres = int(self.img_size/2)
        offset_lowres = int(tile_size_lowres/2)
        self.num_tiles = int((self.observations_highres.shape[2]-tile_size_highres)/offset_highres)*int((self.observations_highres.shape[3]-tile_size_highres)/offset_highres)
        self.predictions = self.get_tiles(self.predictions, tile_size_lowres)
        self.observations = self.get_tiles(self.observations, tile_size_lowres)
        self.observations_highres = self.get_tiles(self.observations_highres, tile_size_highres)
        assert self.predictions.shape[0] == self.observations.shape[0] == self.observations_highres.shape[0]


        ## Are we changing the data by standardizing, should we revert before plotting

    def __len__(self):
        return len(self.predictions)

    def standardize_images(self, image_set):
        #Standardizing to 0.5 mean, 0.5 std
        #image_set -= np.mean(image_set, axis=0, keepdims=True)
        #image_set /= 2.0*np.std(image_set, axis = 0, keepdims=True)
        #image_set += 0.5
        #image_set = np.sqrt(image_set)
        image_set = image_set /10.0
        image_set = np.clip(image_set, a_min=0.0, a_max=1.0)
        image_set = np.expand_dims(image_set, axis=1)
        return image_set
    
    def get_tiles(self, image_set, tile_size):
        offset = int(tile_size/2)
        result = []
        for row in range(0, image_set.shape[2]-tile_size, offset):
            for col in range(0, image_set.shape[3]-tile_size, offset):
                result.append(image_set[:, :, row:row+tile_size, col:col+tile_size])
        result = np.concatenate(result, axis=0)
        return result

    def __getitem__(self, index):
        if self.highres:
            return self.predictions[index], self.observations_highres[index]
        else:
            return self.predictions[index], self.observations[index]

class Dataset(torch.utils.data.Dataset):
    def __init__(self, device='cpu', on_cluster=False):
        self.device = device
        self.selected_indices = []
        self.base_dir = "/mnt/ds3lab-scratch/bhendj/data"
        self.observations = load_observations(self.base_dir, on_cluster)
        self.observations_highres = dict()
        self.cosmo = load_predictions(self.base_dir, on_cluster)
        self.compute_nearest_neighbors()



    def __len__(self):
        return len(self.selected_indices)

    def get_obs_by_id(self, id):
        key, id, reftime, leadtime = id.split("-", maxsplit=3)
        id = int(id)
        return self.observations[key].isel(time=id, dummy=0), self.observations_highres[key].isel(time=id, dummy=0)

    def compute_nearest_neighbors(self):

        chx, chy = self.cosmo['chx'], self.cosmo['chy']
        curr_x, curr_y = chx.x.values, chx.y.values
        ext_x, ext_y = np.random.rand(2*len(curr_x)-1), np.random.rand(2*len(curr_y)-1)
        mid_x, mid_y = (curr_x[1:] + curr_x[:-1])/2.0, (curr_y[1:] + curr_y[:-1])/2.0
        ext_x[::2], ext_y[::2] = curr_x, curr_y
        ext_x[1::2], ext_y[1::2] = mid_x, mid_y
        chx_ext = chx.interp(x=ext_x, y=ext_y)
        chy_ext = chy.interp(x=ext_x, y=ext_y)
        for key, obs in self.observations.items():
            self.observations[key] =  obs.sel(chx=chx, chy=chy, method='nearest')
            self.observations_highres[key] = obs.sel(chx=chx_ext, chy=chy_ext, method='nearest')

    def get_x_y_by_id(self, id: str):
        # TODO: list all times
        t = time.time()
        obs_point, obs_point_highres = self.get_obs_by_id(id)
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
        real_point_highres = obs_point_highres
        prec_pred = pred_points['PREC'].values
        prec_real = real_point['RR'].values
        prec_real_highres = real_point_highres['RR'].values
        if np.any(np.isnan(prec_pred)):
            warnings.warn(f"nan value encountered in ensemble for reftime {reftime} leadtime {leadtime}")

        return prec_pred, prec_real, prec_real_highres  # X: predictions, Y: observations

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

        x, y, y_highres = self.get_x_y_by_id(self.selected_indices[index])
        x = torch.tensor(x, device=self.device)
        y = torch.tensor(y, device=self.device)
        y_highres = torch.tensor(y_highres, device=self.device)
        return x, y, y_highres


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


def create_training_dataset(out_dir="/mnt/ds3lab-scratch/dslab2019/shghosh/preprocessed/"):
    ds = Dataset(device='cpu', on_cluster=True)
    ds.select_all()
    assert len(ds) > 0
    buffer_x, buffer_y, buffer_y_highres = [], [], []
    file_offset = 0

    for i, (x, y, y_highres) in tqdm(enumerate(ds), total=len(ds)):
        if torch.any(x.isnan()) or torch.any(y.isnan()) or torch.any(y_highres.isnan()):
            continue

        buffer_x.append(x)
        buffer_y.append(y)
        buffer_y_highres.append(y_highres)

        if len(buffer_x) > 1024:
            save_buffer_to_file(buffer_x, join(out_dir, f"x.{file_offset}.pt"))
            save_buffer_to_file(buffer_y, join(out_dir, f"y.{file_offset}.pt"))
            save_buffer_to_file(buffer_y_highres, join(out_dir, f"yHighres.{file_offset}.pt"))
            file_offset += 1
            buffer_x, buffer_y, buffer_y_highres = [], [], []

    if len(buffer_x) > 0:
        save_buffer_to_file(buffer_x, join(out_dir, f"x.{file_offset}.pt"))
        save_buffer_to_file(buffer_y, join(out_dir, f"y.{file_offset}.pt"))
        save_buffer_to_file(buffer_y_highres, join(out_dir, f"yHighres.{file_offset}.pt"))


if __name__ == "__main__":
    create_training_dataset()
    # d = Dataset()
    # prec_pred, prec_real = d.get_x_y_at_time(0)
    # plotit()
