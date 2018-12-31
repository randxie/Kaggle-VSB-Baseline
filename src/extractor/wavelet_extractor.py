import numpy as np

from .abstract_extractor import AbstractExtractor
from numba import jit
from utils import util_signal


@jit('float32(float32[:,:])')
def _extract_wavelet_entropy_hjoth(signals):
  signals = util_signal.add_high_pass_filter(signals)
  feature_vec = []
  feature_vec.extend(util_signal.compute_entropy(signals))
  feature_vec.extend(util_signal.compute_wavelet_energy(signals))
  feature_vec.extend(util_signal.compute_hjoth(signals))

  return np.array(feature_vec)


class WaveletExtractor(AbstractExtractor):
  """Extract wavelet energy, signal entropy and Hjoth parameters."""

  def __init__(self, dataloader, folder='', name='wavelet', **extra_params):
    super().__init__(dataloader, folder, name)
    self.process_fn = None
    self.extract_fn = _extract_wavelet_entropy_hjoth
