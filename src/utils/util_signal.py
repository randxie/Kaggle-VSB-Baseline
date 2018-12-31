from common import SAMPLING_FREQ
import numpy as np
import scipy
from scipy import stats
from scipy import signal
from scipy.signal import butter, iirnotch
from scipy.fftpack import fft, ifft
import pywt
from statsmodels.robust import mad


def add_high_pass_filter(signals, low_freq=1000, sample_fs=SAMPLING_FREQ):
  # Fault pattern usually exists in high frequency band. According to literature, the pattern is visible above 10^4 Hz.
  sos = butter(10, low_freq, btype='hp', fs=sample_fs, output='sos')
  filtered_sig = signal.sosfilt(sos, signals)

  return filtered_sig


def rms_normalize_signal(signal):
  return signal / (np.sqrt(np.sum(signal ** 2) / len(signal)))


def wavelet_denoising(signals, threshold=None, wavelet="db4", level=1):
  # calculate the wavelet coefficients
  coeff = pywt.wavedec(signals, wavelet, mode="per")

  # calculate a threshold
  sigma = threshold or mad(coeff[-level])

  # changing this threshold also changes the behavior,
  # but I have not played with this very much
  uthresh = sigma * np.sqrt(2 * np.log(len(signals)))
  coeff[1:] = (pywt.threshold(i, value=uthresh, mode="soft") for i in coeff[1:])
  # reconstruct the signal using the thresholded coefficients
  filtered = pywt.waverec(coeff, wavelet, mode="per")
  return filtered


def compute_wavelet_energy(signals, wavelet="db4"):
  # calculate the wavelet coefficients
  coeffs = pywt.wavedec(signals, wavelet, mode="per")
  energy = []
  for coeff in coeffs:
    coeff_energy = (coeff ** 2)
    energy.append(np.mean(coeff_energy))
    energy.append(np.max(coeff_energy))
    energy.append(np.sum(coeff_energy))
    energy.append(np.sqrt(np.var(coeff_energy)))

  return energy


def compute_entropy(signals):
  histogram = np.histogram(signals, bins=20)[0]
  norm_hist = histogram / np.sum(histogram)
  norm_hist = norm_hist[norm_hist > 0]
  s = - np.sum(norm_hist * np.log2(norm_hist))
  return [s]


def compute_hjoth(signals):
  activity = np.var(signals)
  sig_diff = np.diff(signals)
  mobility = np.sqrt(np.var(sig_diff) / activity)
  mobility_diff = np.sqrt(np.var(np.diff(sig_diff)) / np.var(sig_diff))
  complexity = mobility_diff / mobility
  return [activity, mobility, complexity]
