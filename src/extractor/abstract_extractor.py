import abc
from common import CACHING_DIR
from common import IS_DEBUG
import logging
import numpy as np
import os
import pickle
from torch.utils.data import DataLoader as TorchDataloader


class AbstractExtractor(abc.ABC):
  def __init__(self, dataset, folder='', name=None):
    self._dataset = dataset
    self._folder = folder
    self._name = name
    self._feature = None
    self._signal_id = None
    self.process_fn = None
    self.extract_fn = None

  @property
  def feature(self):
    return self._feature

  @property
  def signal_id(self):
    return self._signal_id

  def process_batch(self, batch_size=12):
    if self._feature is not None:
      return self._feature

    # set feature extraction function
    self._dataset.extract_fn = self.extract_fn
    self._dataset.process_fn = self.process_fn

    num_signals = len(self._dataset)
    try:
      # estimate feature vector dimension using the first signal
      tmp_feature = self._dataset[0]
      num_feat = len(tmp_feature)
    except:
      raise Exception("Failed to extract feature from signals")

    self._feature = np.zeros((num_signals, num_feat))
    self._signal_id = self._dataset.metadata['signal_id'] # for consistency checking when merging different features.

    # set pytorch loader
    num_workers = 4 if IS_DEBUG else 8
    loader = TorchDataloader(self._dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    logging.info("Total number of batches is %s", len(loader))

    # compute feature in batch
    for batch_i, cur_features in enumerate(loader):
      logging.info("In %s feature, processing batch: %d", self._name, batch_i)
      start_idx = batch_i * batch_size
      end_idx = (batch_i + 1) * batch_size
      if end_idx > num_signals:
        end_idx = num_signals

      # load batch data
      self._feature[start_idx:end_idx, :] = cur_features

  @property
  def feature_name(self):
    return "%s_%s" % (self._dataset.mode, self._name)

  def load(self):
    if self._feature is None:
      self._feature, self._signal_id = pickle.load(
        open(os.path.join(CACHING_DIR, self._folder, '%s.p' % self.feature_name), 'rb'))
    else:
      raise Exception("Please instantiate a new object for feature loading.")

  def save(self):
    pickle.dump([self._feature, self._signal_id],
                open(os.path.join(CACHING_DIR, self._folder, '%s.p' % self.feature_name), 'wb'))
