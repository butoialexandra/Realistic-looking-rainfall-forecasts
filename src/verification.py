import numpy as np
from pysteps import verification

def crps(pred, obs):
    crps = verification.probscores.CRPS_init()
    verification.probscores.CRPS_accum(crps, pred, obs)
    return verification.probscores.CRPS_compute(crps)

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

