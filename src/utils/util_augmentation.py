import numpy as np

NUM_SIGNAL_LEN = 80000


def augment_signal(metadata_to_aug,
                   min_loc=NUM_SIGNAL_LEN // 16,
                   max_loc=NUM_SIGNAL_LEN // 2,
                   seed=4242):
  """Generate augmented signal metadata.

  :param metadata_to_aug:  Signal to augment, currently only augment signal with PD pattern.
  :param min_loc: min singal indicies to do the shift
  :param max_loc: max singal indicies to do the shift
  :param seed: a random seed for reproducibility
  :return: A pandas dataframe
  """
  # set seed for reproducibility
  np.random.seed(seed)
  augmented = metadata_to_aug.copy()
  num_aug = augmented.shape[0]

  # update signal id name. keep the group and phase the same, which will be used in cross validation.
  augmented['signal_id'] = augmented['signal_id'].apply(lambda x: str(x) + '__' + str(seed))

  # assign shift index
  shift_index = np.random.randint(min_loc, max_loc, size=num_aug)
  augmented['shift_idx'] = shift_index

  return augmented


def shift_and_concat(signal, shift_idx):
  """Shift and concatenate signal based on shift index

  :param signal: A numpy array
  :param shift_idx: An integer representing shift index
  :return: A numpy array
  """
  if shift_idx == 0:
    return signal

  aug_sig = np.zeros_like(signal)
  num_points = len(aug_sig)
  aug_sig[0:(num_points-shift_idx)] = signal[shift_idx:num_points]
  aug_sig[(num_points-shift_idx):num_points] = signal[0:shift_idx]

  return aug_sig
