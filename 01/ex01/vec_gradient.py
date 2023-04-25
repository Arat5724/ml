import numpy as np
from numpy import ndarray


def add_intercept(x: ndarray) -> ndarray:
    try:
        if x.ndim == 1:
            x = x.reshape((*x.shape, 1))
        m, n = x.shape
        if m * n == 0:
            return None
        return np.hstack((np.ones((m, 1)), x))
    except Exception as e:
        print(e)


def gradient(x: ndarray, y: ndarray, theta: ndarray) -> ndarray:
    """
    Computes a gradient vector from three non-empty numpy.array, without any for loop.
        The three arrays must have compatible shapes.
    Args:
        x: has to be a numpy.array, a matrix of shape m * 1.
        y: has to be a numpy.array, a vector of shape m * 1.
        theta: has to be a numpy.array, a 2 * 1 vector.
    Return:
        The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
        None if x, y, or theta is an empty numpy.ndarray.
        None if x, y and theta do not have compatible dimensions.
    Raises:
        This function should not raise any Exception.
    """

    x = add_intercept(x)
    if y.ndim == 1:
        y = y.reshape((*y.shape, 1))
    if theta.ndim == 1:
        theta = theta.reshape((*theta.shape, 1))
    m, _ = y.shape
    y_hat = np.dot(x, theta)
    return np.dot(x.T, y_hat - y).reshape((2, 1)) / m
