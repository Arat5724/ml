import numpy as np
from numpy import ndarray


def sigmoid_(x):
    return 1 / (1 + np.exp(-x))


def vec_log_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.ndarray, with a for-loop. The three arrays must have compatiblArgs:
        x: has to be an numpy.ndarray, a matrix of shape m * n.
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        theta: has to be an numpy.ndarray, a vector of shape (n + 1) * 1.
    Returns:
        The gradient as a numpy.ndarray, a vector of shape n * 1, containing the result of the formula for all j.
        None if x, y, or theta are empty numpy.ndarray.
        None if x, y and theta do not have compatible dimensions.
    Raises:
        This function should not raise any Exception.
    """
    m, n = x.shape
    x1 = np.hstack((np.ones((m, 1)), x))
    y_hat = sigmoid_(np.dot(x1, theta))
    return np.dot(x1.T, y_hat - y) / m
