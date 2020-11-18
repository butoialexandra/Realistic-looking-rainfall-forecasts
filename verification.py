
import numpy as np
import properscoring as ps

def crps(pred, obs):
    crps = verification.probscores.CRPS_init()
    verification.probscores.CRPS_accum(crps, pred, obs)
    return verification.probscores.CRPS_compute(crps)

def log_spectral_distance(pred, obs):
    def power_spectrum_dB(img):
        fx = np.fft.fft2(img)
        fx = fx[:img.shape[0]//2,:img.shape[1]//2]
        px = abs(fx)**2
        return 10 * np.log10(px)

    d = (power_spectrum_dB(obs)-power_spectrum_dB(pred))**2

    d[~np.isfinite(d)] = np.nan
    return np.sqrt(np.nanmean(d))
