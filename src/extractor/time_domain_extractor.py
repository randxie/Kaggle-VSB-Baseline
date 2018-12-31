import numpy as np

from .abstract_extractor import AbstractExtractor
from numba import jit
from scipy import stats
from scipy.signal import find_peaks
from utils import util_signal


@jit('float32(float32[:,:])')
def _denoise_signal(signals):
  # suppress strong sine pattern
  filtered_sig = util_signal.add_high_pass_filter(signals)
  filtered_sig = util_signal.wavelet_denoising(filtered_sig, threshold=1)
  return filtered_sig


@jit('float32(float32[:,:])')
def _extract_time_feature(signals):
  # basic time domain statistics
  sig_mean = np.mean(signals)
  sig_std = np.sqrt(np.var(signals))
  sig_skew = stats.skew(signals)
  sig_kurt = stats.kurtosis(signals)
  sig_abs_max = np.max(np.abs(signals))

  # compute peak statistics based on paper "A Complex Classification Approach of Partial
  # Discharges from Covered Conductors in Real Environment"
  peaks, _ = find_peaks(signals, height=np.max(signals) / 6, distance=10)
  sig_peak = signals[peaks]
  peak_diff = np.diff(peaks)

  if len(peak_diff) == 0:
    mean_peak_width = np.nan
    max_peak_width = np.nan
    min_peak_width = np.nan
  else:
    mean_peak_width = np.mean(peak_diff)
    max_peak_width = np.max(peak_diff)
    min_peak_width = np.min(peak_diff)

  num_peaks = len(peaks)
  mean_peak_height = np.mean(sig_peak)
  max_peak_height = np.max(sig_peak)
  min_peak_height = np.min(sig_peak)

  return np.array([sig_mean, sig_std, sig_skew, sig_kurt, sig_abs_max,
                   num_peaks, mean_peak_width, mean_peak_height, max_peak_width, max_peak_height,
                   min_peak_width, min_peak_height])


class TimeDomainExtractor(AbstractExtractor):
  """Extract basic time domain statistics."""

  def __init__(self, dataloader, folder='', name='time_domain', **extra_params):
    super().__init__(dataloader, folder, name)
    self.process_fn = _denoise_signal
    self.extract_fn = _extract_time_feature
