import logging


def set_logging_config(log_file_path):
  """Set up logging.

  :param log_file_path: Log file location.
  :return: None
  """
  logging.basicConfig(level=logging.DEBUG,
                      format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                      datefmt='%m-%d %H:%M',
                      filename=log_file_path,
                      filemode='w')

  # output logging info to screen as well (from python official website)
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
  console.setFormatter(formatter)
  logging.getLogger('').addHandler(console)
