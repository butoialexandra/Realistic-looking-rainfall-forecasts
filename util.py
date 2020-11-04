import pyproj
import xarray as xr
import matplotlib.pyplot as plt
import time


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