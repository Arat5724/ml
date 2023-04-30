import numpy as np
from numpy import ndarray


def sigmoid_(x):
    return 1 / (1 + np.exp(-x))


def logistic_predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * n.
        theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
    Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
        None if x or theta are empty numpy.ndarray.
        None if x or theta dimensions are not appropriate.
    Raises:
        This function should not raise any Exception.
    """
    m, n = x.shape
    x1 = np.hstack((np.ones((m, 1)), x))
    return sigmoid_(np.dot(x1, theta))
