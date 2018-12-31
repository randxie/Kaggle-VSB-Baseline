from sklearn.metrics import matthews_corrcoef


def compute_score(y_true, y_pred):
  """Metric used in the Kaggle VSB competition.

  :param y_true: Truth label
  :param y_pred: Predicted label
  :return: Score value
  """
  return matthews_corrcoef(y_true, y_pred)