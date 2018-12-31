import os
import matplotlib
matplotlib.use('Agg')

# Whether to run debug mode (use only 100 signals to test pipeline integrity)
IS_DEBUG = False
if IS_DEBUG:
  print("Running in debug mode.")
else:
  print("Running in prod mode")

# Max number of signals to store in the data loader.
MAX_CACHE_SIZE = 2000

# Edit settings here
PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(PROJECT_PATH, '..')
DATA_DIR = os.path.join(ROOT_DIR, 'input')
CACHING_DIR = os.path.join(ROOT_DIR, 'caching') # cached feature matrices
RESULTS_DIR = os.path.join(ROOT_DIR, 'result') # logging, models, etc

os.makedirs(CACHING_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Signal characteristics
SAMPLING_FREQ = 80000/0.02 # 80,000 data points taken over 20 ms
