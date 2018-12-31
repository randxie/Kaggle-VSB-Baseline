import logging
import numpy as np
import os
import pandas as pd
import pyarrow.parquet as parq

from common import DATA_DIR
from common import IS_DEBUG
from common import MAX_CACHE_SIZE
from torch.utils.data import Dataset as TorchDataset
from utils import util_augmentation
from utils import util_plot
from utils import util_signal

AUGMENTATION_SEED = [666, 1234, 8888]


class VsbSignalDataset(TorchDataset):

  def __init__(self, mode='train', preprocess_fn=None, extract_fn=None, use_augmentation=False, use_cache=False):
    self.filename = os.path.join(DATA_DIR, '%s.parquet' % mode)
    self.metadata = pd.read_csv(os.path.join(DATA_DIR, 'metadata_%s.csv' % mode))
    self.use_cache = use_cache
    self.use_augmentation = use_augmentation

    if IS_DEBUG:
      # use first 10 rows in the debug mode
      self.metadata = self.metadata.loc[0:100, :]

    if use_augmentation and mode == 'train':
      self.metadata['shift_idx'] = 0
      pos_meta_data = self.metadata.loc[self.metadata.target == 1, :].copy()

      for aug_seed in AUGMENTATION_SEED:
        # augment signal by random shifting and concatenation
        cur_augmented = util_augmentation.augment_signal(pos_meta_data, seed=aug_seed)
        self.metadata = pd.concat([self.metadata, cur_augmented], axis=0)

      self.metadata.reset_index(drop=True, inplace=True)

    if use_cache:
      self._cache = {}

    self.mode = mode
    self.preprocess_fn = preprocess_fn
    self.extract_fn = extract_fn

  def __getitem__(self, index):
    """Get signals given index.

    It does not assume the signals have the same size, as the signal might be truncated.

    :param index: An integer
    :return: A signal in numpy array
    """
    if self.use_cache and index in self._cache:
      return self._cache[index]

    col_to_load = str(self.metadata['signal_id'].loc[index])
    col_to_load = col_to_load.split('__')[0]

    # read data from parquet and do the conversion
    signal = parq.read_table(self.filename, columns=[col_to_load]).to_pydict()[col_to_load]
    signal = np.array(signal)

    if self.use_augmentation and self.mode == 'train':
      signal = util_augmentation.shift_and_concat(signal, int(self.metadata['shift_idx'].loc[index]))

    if self.preprocess_fn:
      signal = self.preprocess_fn(signal)

    if self.extract_fn:
      signal = self.extract_fn(signal)

    if self.use_cache and len(self._cache) < MAX_CACHE_SIZE:
      # simple caching mechanism should work ok here, as pytorch requires random access during training.
      self._cache[index] = signal

    return signal

  def __len__(self):
    return self.metadata.shape[0]

  @property
  def labels(self):
    return self.metadata['target'].values

  @property
  def groups(self):
    return self.metadata['id_measurement'].values


if __name__ == '__main__':
  data = VsbSignalDataset(mode='train')
  idx = 3
  signal = data[idx]
  filtered_sig = util_signal.add_high_pass_filter(signal)
  filtered_sig = util_signal.wavelet_denoising(filtered_sig, threshold=1)

  util_plot.plot_fft_with_signal(filtered_sig)
  util_plot.plot_filtered_with_signal(filtered_sig, signal)
  util_plot.plot_stft_with_signal(filtered_sig)
  util_plot.plot_peaks_with_signal(filtered_sig)
