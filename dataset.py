
import torch

import os
from os.path import join
import numpy as np
import xarray as xr
import pyproj


base_dir = "/mnt/ds3lab-scratch/bhendj/data"
on_cluster = True




def load_predictions():
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


def load_observations_and_predictions():
    predictions = load_predictions()
    if on_cluster:
        path = join(base_dir, "combiprecip", "combiprecip_201805.nc")
    else:
        path = "./combiprecip_201805.nc"
    weather = xr.open_mfdataset(path, combine='by_coords')
    weather_itv = weather.isel(time=slice(0,-1,12))
    target_time =  weather_itv['time'].values
    predictions_itv = predictions.sel(reftime=target_time-np.timedelta64(1, 'h')).isel(leadtime=1, member=0)
    chx, chy = predictions.chx, predictions.chy
    real_points = weather_itv.sel(chx=chx, chy=chy, method='nearest')
    weather_images_numpy =  real_points['RR'].values
    pred_images_numpy =  predictions_itv['PREC'].values
    mask = (weather_images_numpy == 0.0)
    rain_nil_percentage = mask.sum((1,2))/np.prod(mask[0].shape)
    weather_selected = weather_images_numpy[rain_nil_percentage < 0.9, :, :]
    preds_selected = pred_images_numpy[rain_nil_percentage < 0.9, :, :]
    print("Observed shape: " ,weather_selected.shape)
    print("Predictions shape: ", preds_selected.shape)
    return weather_selected, preds_selected



#class Dataset(torch.utils.data.Dataset):
#    def __init__(self, device='cpu'):
#        self.device = device
#        self.observations = load_observations()
#        self.cosmo = load_predictions()
#        self.ids = []
#    
#    def __len__(self):
#        return 128 #len(self.ids)
#
#    def get_x_y_at_time(self, target_time: int):
#        # TODO: list all times
#        obs_point = self.observations.isel(time=target_time, dummy=0)
#        target_time = obs_point['time'].values
#        pred_points = self.cosmo.sel(reftime=target_time - np.timedelta64(1, 'h')).isel(leadtime=1, member=0)  # TODO: member!
#        
#        # nearest neighbors lookup
#        chx, chy = pred_points['chx'], pred_points['chy']
#        #chx.values, chy.values
#        real_point = obs_point.sel(chx=chx, chy=chy, method='nearest')
#        #real_point['chx'].values, real_point['chy'].values
#        prec_pred = pred_points['PREC'].values
#        prec_real = real_point['RR'].values
#        print("Shapes:", prec_real.shape, prec_pred.shape)
#        return prec_pred, prec_real  # X: predictions, Y: observations
#
#    def __getitem__(self, index):
#        flag = True
#        curr_idx =  index
#        while (flag == True):
#            import pdb; pdb.set_trace()
#            x, y = self.get_x_y_at_time(curr_idx)
#            x = torch.tensor(x, device=self.device)
#            y = torch.tensor(y, device=self.device)
#            if ((y == 0.0).sum()/y.size > 0.9):
#                curr_idx += 12
#            else:
#                flag = False
#        return x, y
#


class Dataset(torch.utils.data.Dataset):
    def __init__(self, device = 'cpu'):
        self.device = device
        self.observations, self.predictions = load_observations_and_predictions()

    def __len__(self):
        return self.observations.shape[0]

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index =  index.to_list()
        x, y = self.observations[index], self.predictions[index]
        x = torch.tensor(x, device=self.device)
        y = torch.tensor(y, device=self.device)
        return x,y


#def plotit():
#    import numpy as np
#    import matplotlib.pyplot as plt
#    
#    prec_pred, prec_real = d.get_x_y_at_time(0)
#
#    fig = plt.figure(figsize=(6, 3.2))
#
#    ax = fig.add_subplot(121)
#    ax.set_title('Prediction #0')
#    plt.imshow(prec_pred)
#    # ax.set_aspect('equal')
#
#    ax = fig.add_subplot(122)
#    ax.set_title('Observation')
#    plt.imshow(prec_real)
#    # ax.set_aspect('equal')
#
#    # cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
#    # cax.get_xaxis().set_visible(False)
#    # cax.get_yaxis().set_visible(False)
#    # cax.patch.set_alpha(0)
#    # cax.set_frame_on(False)
#    plt.colorbar(orientation='horizontal')
#    plt.savefig("foo.png")
#

if __name__ == "__main__":

    d = Dataset()
    #prec_pred, prec_real = d.get_x_y_at_time(0)
    #plotit()
