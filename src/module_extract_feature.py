import argparse
import logging
import os

from common import CACHING_DIR
from dataset.vsb_signal_dataset import VsbSignalDataset
from extractor.time_domain_extractor import TimeDomainExtractor
from extractor.wavelet_extractor import WaveletExtractor
from utils import util_config
from utils import util_logging

FEATURE_OBJ_MAP = {
  'time_stat': TimeDomainExtractor,
  'wavelet': WaveletExtractor,
}


def _init_env(folder_name):
  experiment_folder = os.path.join(CACHING_DIR, folder_name)
  os.makedirs(experiment_folder, exist_ok=True)

def _init_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--config')
  parser.add_argument('--mode', default='train')
  return parser


def extract_feature(config, mode):
  _init_env(config['name'])
  feat_config = config['feature']
  feat_list = []
  for feat_name, feat_setting in feat_config.items():
    logging.info('Extracting feature: %s', feat_name)
    feat_extractor_fn = FEATURE_OBJ_MAP[feat_name]
    vsb_dataset = VsbSignalDataset(mode=mode)
    feat_extractor = feat_extractor_fn(vsb_dataset, folder=config['name'])
    try:
      feat_extractor.load()
    except Exception as e:
      logging.warning('Failed to extract features with error: %s. Will try to re-generate features.', e)
      feat_extractor.process_batch()
      feat_extractor.save()

    feat_list.append(feat_extractor.feature)

  return feat_list


if __name__ == '__main__':
  parser = _init_parser()
  args = parser.parse_args()

  # extract features
  mode = args.mode
  config = util_config.load_config(args.config)

  experiment_folder = os.path.join(CACHING_DIR, config['name'])
  util_logging.set_logging_config(os.path.join(experiment_folder, 'feature.log'))
  extract_feature(config, mode)
