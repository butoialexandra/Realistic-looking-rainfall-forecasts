import pyproj
import xarray as xr

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


if __name__ == '__main__':
    cosmo = xr.open_zarr("cosmoe_prec_201805.zarr")
    cosmo = reproject_to_swiss_coords(cosmo)
    print(cosmo)