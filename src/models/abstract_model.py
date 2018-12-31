import abc
import logging
import numpy as np
import os
import pickle

from common import RESULTS_DIR
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import GroupKFold


class AbstractModel(abc.ABC):
  def __init__(self, features, labels, params, model_name=None, folder_name='', task='classification', **extra_params):
    self.features = features
    self.labels = labels
    self.params = params
    self.model_name = model_name
    self.folder_name = folder_name
    self.task = task
    self.extra_params = extra_params

    self.initialize()

  @abc.abstractmethod
  def initialize(self):
    ...

  # enumulate sklearn interface
  def fit(self, X, y):
    logging.info("start training")
    self._train(X, y)

  def predict(self, X):
    return self._predict(X)

  # for cross validation and stacking
  def cvtrain(self, train_idx):
    self.initialize()
    self._train(self.features.loc[train_idx, :], self.labels[train_idx])

  def cvpredict(self, test_idx):
    test_in = self.features.loc[test_idx, :]
    return self._predict(test_in)

  # ------------------------
  # File IO related
  # ------------------------
  def is_file_exist(self):
    p_file = os.path.join(RESULTS_DIR, self.folder_name, '%s.p' % self.model_name)
    return os.path.isfile(p_file)

  def save(self):
    p_file = os.path.join(RESULTS_DIR, self.folder_name, '%s.p' % self.model_name)
    # save file to pickle
    if not os.path.isfile(p_file):
      logging.info('File %s does not exist. Will create the p-file' % p_file)

    with open(p_file, 'wb') as f:
      pickle.dump(self.mdl, f, pickle.HIGHEST_PROTOCOL)

  def load(self):
    p_file = os.path.join(RESULTS_DIR, self.folder_name, '%s.p' % self.model_name)
    if os.path.isfile(p_file):
      with open(p_file, 'rb') as f:
        self.mdl = pickle.load(f)
    else:
      logging.info('Can not find file %s' % p_file)
      raise FileNotFoundError(p_file)

  # -------------------------------
  # Internal train/predict methods
  # -------------------------------
  def _train(self, X, y):
    self.initialize()
    self.mdl.fit(X, y)

  # prediction with probability outcome
  def _predict(self, X):
    if "predict_proba" in dir(self.mdl) and self.task == 'classification':
      pred = self.mdl.predict_proba(X)
      pred = pred[:, 1]
    else:
      pred = self.mdl.predict(X)

    return pred

  def eval_model(self, groups=None, score_fn=matthews_corrcoef):
    scores = []
    group_k_fold = GroupKFold(n_splits=4)

    # simulate 1:3 train:test ratio
    for val_idx, train_idx in group_k_fold.split(self.features, groups=groups):
      self.initialize()

      x_train = self.features[train_idx, :]
      y_train = self.labels[train_idx]
      self.fit(x_train, y_train)

      pos_label_ratio = np.sum(y_train) / y_train.shape[0]

      x_val = self.features[val_idx, :]
      y_val = self.labels[val_idx]
      y_pred = self.predict(x_val)

      final_pred = (y_pred > pos_label_ratio).astype(np.int)
      scores.append(score_fn(y_val, final_pred))

    logging.info('Model evaluation results: %s', scores)
    logging.info('Mean score: %f', np.mean(scores))
    logging.info('Std score: %f', np.sqrt(np.var(scores)))
    return scores
