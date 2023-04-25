import numpy as np
from numpy import ndarray


def gradient(x: ndarray, y: ndarray, theta: ndarray) -> ndarray:
    """Computes a gradient vector from three non-empty numpy.array, without any for-loop.
        The three arrays must have the compatible dimensions.
    Args:
        x: has to be an numpy.array, a matrix of dimension m * n.
        y: has to be an numpy.array, a vector of dimension m * 1.
        theta: has to be an numpy.array, a vector (n + 1) * 1.
    Return:
        The gradient as a numpy.array, a vector of dimensions n * 1,
        containg the result of the formula for all j.
        None if x, y, or theta are empty numpy.array.
        None if x, y and theta do not have compatible dimensions.
        None if x, y or theta is not of expected type.
    Raises:
        This function should not raise any Exception.
    """

    m, _ = x.shape
    x = np.hstack((np.ones((m, 1)), x))
    y_hat = np.dot(x, theta)
    return np.dot(x.T, y_hat - y) / m
