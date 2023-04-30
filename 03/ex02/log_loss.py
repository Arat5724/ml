import numpy as np
from numpy import ndarray


def sigmoid_(x):
    return 1 / (1 + np.exp(-x))


def logistic_predict_(x, theta):
    m, n = x.shape
    x1 = np.hstack((np.ones((m, 1)), x))
    return sigmoid_(np.dot(x1, theta))


def log_loss_(y, y_hat, eps=1e-15):
    """
    Computes the logistic loss value.
    Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        eps: has to be a float, epsilon (default=1e-15)
    Returns:
        The logistic loss value as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    m, n = y.shape
    return (y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)).sum() / -m
