import numpy as np
from pysteps import verification

def crps_ensemble(observation, forecasts):
    """
    Computes per image continuous ranked probability score between
    an observation and a probabilistic forecast; code adapted from
    https://github.com/jleinonen/downscaling-rnn-gan/blob/master/dsrnngan/crps.py
    :param observation: observation shaped (height, width)
    :param forecasts: forecast shaped (height, width, members)
    :return: per image CRPS
    """
    fc = forecasts.copy()
    fc.sort(axis=-1)
    obs = observation
    crps = np.zeros_like(obs)
    obs = np.expand_dims(obs, axis=0)
    fc_below = fc<obs
    obs = observation

    for i in range(fc.shape[0]):
        below = fc_below[i,:,:]
        weight = ((i+1)**2 - i**2) / fc.shape[-1]**2
        crps[below] += weight * (obs[below]-fc[i,:,:][below])

    for i in range(fc.shape[0]-1,-1,-1):
        above  = ~fc_below[i,:,:]
        k = fc.shape[-1]-1-i
        weight = ((k+1)**2 - k**2) / fc.shape[-1]**2
        crps[above] += weight * (fc[i,:,:][above]-obs[above])

    return np.mean(crps)


def log_spectral_distance(pred, obs):
    """
    Computes log spectral distance between one observation and
    one prediction
    :param pred: generated prediction
    :param obs: observation
    :return: log spectral distance
    """
    d = (power_spectrum_dB(obs)-power_spectrum_dB(pred))**2
    d[~np.isfinite(d)] = np.nan
    return np.sqrt(np.nanmean(d))

def power_spectrum_dB(img):
    """
    Computes power spectrum of an image
    :param img: 2d image
    :return: 2d power spectrum
    """
    fx = np.fft.fft2(img)
    #fx = fx[:img.shape[0]//2,:img.shape[1]//2]
    px = abs(fx)**2
    px =  10 * np.log10(px)
    # px[~np.isfinite(px)] = np.nan

    return px

def power_spectrum_batch_avg(imgs):
    """
    Computes the average power spectrum of a batch of images
    :param imgs: batch of images of size (batch size, width, height)
    :return: average power spectrum
    """
    power_spectrum = np.empty_like(imgs)
    batch_size = imgs.shape[0]
    for i in range(0, batch_size):
        power_spectrum[i,:,:] = power_spectrum_dB(imgs[i,:,:])
        #power_spectrum[i,:,:][~np.isfinite(power_spectrum[i,:,:])] = np.nan
    return np.nanmean(power_spectrum, axis=0)

def log_spectral_distance_batch(preds, obs):
    """
    Computes the log spectral distance between the average power spectrum of
    predictions and average power spectrum of observations
    :param preds: batch of predictions
    :param obs: batch of observations
    :return: log spectral distance between the average power spectrum of
    predictions and average power spectrum of observations
    """
    d = (power_spectrum_batch_avg(obs) - power_spectrum_batch_avg(preds)) ** 2
    d[~np.isfinite(d)] = np.nan
    return np.sqrt(np.nanmean(d))

def log_spectral_distance_pairs_avg(preds, obs):
    """
    Computes average log spectral distance between pairs of predictions
    and observations
    :param preds: a batch of generated predictions
    :param obs: a batch of observations
    :return: average log spectral distance between pairs of predictions
    and observations
    """
    assert preds.shape == obs.shape, "Dimensions of preds and obs must match"
    lsds = np.empty(preds.shape[0])
    for i in range(preds.shape[0]):
        lsds[i] = log_spectral_distance(preds[i,:,:], obs[i,:,:])
    return np.nanmean(lsds)

def lsd_radial(gen_ensemble, obs):
    num_members = gen_ensemble.shape[0]
    power_spectrum_gen  = np.empty(gen_ensemble.shape)
    dist = (power_spectrum_dB(gen_ensemble) - power_spectrum_dB(obs))**2
    dist[~np.isfinite(dist)] = np.nan
    return np.sqrt(np.nanmean(dist, axis=0))




def rmse(generated, real):
    """
    Computes root mean squared error between a batch of generated
    and a batch of real images
    :param generated: batch of generated imgs
    :param real: batch of real imgs
    :return: RMSE
    """
    return np.sqrt(((generated-real)**2).mean())
