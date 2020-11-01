import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from util import reproject_to_swiss_coords, get_nn

from pysteps.postprocessing import ensemblestats
from pysteps import verification

import properscoring as ps


def reliability_diagram(forecast, obs, threshold):
    probs = ensemblestats.excprob(forecast, threshold, ignore_nan=True)
    reldiag = verification.reldiag_init(threshold)
    verification.reldiag_accum(reldiag, probs, obs.values)

    fig, ax = plt.subplots()
    verification.plot_reldiag(reldiag, ax)
    ax.set_title("Reliability diagram (+1 hour)")
    plt.show()


def rank_histogram(forecast, obs, threshold):
    n_members = forecast.values.shape[0]
    rankhist = verification.rankhist_init(n_members, threshold)
    verification.rankhist_accum(rankhist, forecast.values, obs.values)

    fig, ax = plt.subplots()
    verification.plot_rankhist(rankhist, ax)
    ax.set_title("Rank histogram (+1 hour)")
    plt.show()

def crps(forecast, obs):
    crps = verification.probscores.CRPS_init()
    verification.probscores.CRPS_accum(crps, forecast.values, obs.values)
    return verification.probscores.CRPS_compute(crps)

def brier_score(forecast, obs, threshold):
    threshold_scores = ps.threshold_brier_score(obs.values, forecast.values.transpose(1, 2, 0),
                                                threshold=threshold)

    return threshold_scores.mean()

def log_spectral_distance(img1, img2):
    def power_spectrum_dB(img):
        fx = np.fft.fft2(img)
        fx = fx[:img.shape[0]//2,:img.shape[1]//2]
        px = abs(fx)**2
        return 10 * np.log10(px)

    d = (power_spectrum_dB(img1)-power_spectrum_dB(img2))**2

    d[~np.isfinite(d)] = np.nan
    return np.sqrt(np.nanmean(d))


if __name__ == '__main__':
    cosmo = xr.open_zarr("cosmoe_prec_201805.zarr")
    cosmo = reproject_to_swiss_coords(cosmo)
    ds = xr.open_mfdataset("combiprecip_201805.nc", combine='by_coords')

    forecast = cosmo.isel(reftime=5, leadtime=1)
    observations = ds.isel(time=5, dummy=0)
    real_points = get_nn(forecast, observations)

    # reliability_diagram(forecast.PREC, real_points.RR, 0.1)
    # rank_histogram(forecast.PREC, real_points.RR, 0.1)
    print(crps(forecast.PREC, real_points.RR))
    print(brier_score(forecast.PREC, real_points.RR, 0.1))