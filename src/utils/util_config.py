import json


def load_config(config_name):
  """Read config in json format.

  :param config_name: Config file location.
  :return: A dict storing pipeline parameters.
  """
  with open(config_name, 'r') as f:
    config = json.load(f)

  return config
