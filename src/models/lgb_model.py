from .abstract_model import AbstractModel
from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor


class LgbModel(AbstractModel):
  def __init__(self, features, labels, params, model_name='lgb', folder_name='', task='classification', **extra_params):
    super().__init__(features, labels, params, model_name=model_name, folder_name=folder_name, task=task,
                     **extra_params)

  def initialize(self):
    if self.task == 'classification':
      self.mdl = LGBMClassifier(**self.params)
    else:
      self.mdl = LGBMRegressor(**self.params)
