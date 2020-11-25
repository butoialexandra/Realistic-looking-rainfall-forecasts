import torch
from torch.utils.data import Dataset

from os.path import join
import xarray as xr
import pyproj

import matplotlib.pyplot as plt

class WGANDataset(Dataset):
    "Wasserstein GAN dataset."
    def __init__(self, on_cluster=False, device="cpu"):
        self.base_dir = "/mnt/ds3lab-scratch/bhendj/data"
        self.on_cluster = on_cluster
        self.device = device
        self.observations = self.load_observations()
        self.predictions = self.load_predictions()
        # print("Loaded data")
        self.compute_nearest_neighbors()
        # print("Computed nearest neighbours")

    def __len__(self):
        return self.observations.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.observations[idx,:,:], device=self.device)

    def load_observations(self):
        if self.on_cluster:
            path = join(self.base_dir, "combiprecip", "combiprecip_201805.nc")
        else:
            path = "../combiprecip_201805.nc"
        weather = xr.open_mfdataset(path, combine='by_coords')
        return weather

    def load_predictions(self):
        if self.on_cluster:
            path = join(self.base_dir, "cosmoe", "data.zarr", "data_ethz.zarr")
        else:
            path = "../cosmoe_prec_201805.zarr"

        cosmo = xr.open_zarr(path)

        # Transform to the other coordinate system
        src_proj = pyproj.Proj("EPSG:4326") # WSG84
        dst_proj = pyproj.Proj("EPSG:21781") # CH1903 / LV03
        src_x = cosmo.lon.values
        src_y = cosmo.lat.values
        dst_x, dst_y = pyproj.transform(src_proj, dst_proj, src_x, src_y, always_xy=True)
        cosmo = cosmo.assign_coords({"chx": (("y", "x"), dst_x) , "chy": (("y", "x"), dst_y)})
        return cosmo

    def compute_nearest_neighbors(self):
        chx, chy = self.predictions['chx'], self.predictions['chy']
        self.observations = self.observations.sel(chx=chx, chy=chy, method='nearest').RR.values

if __name__ == "__main__":
    dataset = WGANDataset()
    for i in range(5):
        sample = dataset[i]
        plt.pcolormesh(sample)
        plt.show()
