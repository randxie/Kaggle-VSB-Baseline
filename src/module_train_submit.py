import argparse
import logging
import numpy as np
import os
import pandas as pd

from common import RESULTS_DIR
from dataset.vsb_signal_dataset import VsbSignalDataset
from module_extract_feature import extract_feature
from models.lgb_model import LgbModel
from utils import util_config
from utils import util_logging


def _init_env(folder_name):
  experiment_folder = os.path.join(RESULTS_DIR, folder_name)
  os.makedirs(experiment_folder, exist_ok=True)
  util_logging.set_logging_config(os.path.join(experiment_folder, 'train_submit.log'))


def _init_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--config')
  return parser


def _output_submission(preds, test_ids, filename):
  out_dict = {}
  out_dict['signal_id'] = test_ids
  out_dict['target'] = preds

  out_df = pd.DataFrame.from_dict(out_dict)
  out_df[['signal_id', 'target']].to_csv(filename, index=None)


def run_train_submit(config):
  logging.info('Run train and submit pipeline.')

  # extract training features
  train_dataloader = VsbSignalDataset(mode='train')
  train_feat_list = extract_feature(config, 'train')
  train_feat = np.hstack(train_feat_list)
  logging.info('Training features have shape: %s.', train_feat.shape)

  # analyze labels
  labels = train_dataloader.labels
  pos_label_ratio = np.sum(labels) / len(labels)
  logging.info('Positive label ratio is %s.' % pos_label_ratio)

  # do the actual training
  params = config['model']['lgb']['param']
  mdl = LgbModel(train_feat, labels, params, folder_name=config['name'])

  logging.info('Running model evaluation.')
  _ = mdl.eval_model(groups=train_dataloader.groups)

  # run full training
  logging.info('Running training on whole dataset.')
  mdl.initialize()
  mdl.fit(train_feat, labels)

  # generate submission
  test_feat_list = extract_feature(config, 'test')
  test_feat = np.hstack(test_feat_list)
  test_dataloader = VsbSignalDataset(mode='test')

  test_probs = mdl.predict(test_feat)
  test_ids = test_dataloader.metadata.signal_id.values
  test_preds = (test_probs > pos_label_ratio).astype(np.int)

  logging.info('Predicted positive label ratio is %s.' % (np.sum(test_preds) / len(test_preds)))
  _output_submission(test_preds, test_ids,
                     os.path.join(RESULTS_DIR, config['name'], 'submission_%s.csv' % config['name']))


if __name__ == '__main__':
  parser = _init_parser()
  args = parser.parse_args()
  config = util_config.load_config(args.config)
  _init_env(config['name'])
  run_train_submit(config)
