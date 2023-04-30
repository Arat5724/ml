import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame, Series

import time


def confusion_matrix_(y_true: ndarray, y_hat: ndarray, labels=None, df_option=False):
    """
    Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
        y:a numpy.array for the correct labels
        y_hat:a numpy.array for the predicted labels
        labels: optional, a list of labels to index the matrix.
        This may be used to reorder or select a subset of labels. (default=None)
        df_option: optional, if set to True the function will return a pandas DataFrame
        instead of a numpy array. (default=False)
    Return:
        The confusion matrix as a numpy array or a pandas DataFrame according to df_option value.
        None if any error.
    Raises:
        This function should not raise any Exception.
    """
    y_true, y_hat = y_true.reshape(-1), y_hat.reshape(-1)
    if labels is None:
        labels = list(set(y_true.tolist() + y_hat.tolist()))
        labels.sort()
    t = np.stack([y_true == t for t in labels], dtype=int)
    p = np.stack([y_hat == p for p in labels], dtype=int)
    if df_option:
        return DataFrame(data=np.dot(t, p.T), index=labels, columns=labels)
    return np.dot(t, p.T)
