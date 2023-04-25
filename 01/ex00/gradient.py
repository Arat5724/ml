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


def simple_gradient(x: ndarray, y: ndarray, theta: ndarray) -> ndarray:
    """
    Computes a gradient vector from three non-empty numpy.array, with a for-loop.
        The three arrays must have compatible shapes.
    Args:
        x: has to be an numpy.array, a vector of shape m * 1.
        y: has to be an numpy.array, a vector of shape m * 1.
        theta: has to be an numpy.array, a 2 * 1 vector.
    Return:
        The gradient as a numpy.array, a vector of shape 2 * 1.
        None if x, y, or theta are empty numpy.array.
        None if x, y and theta do not have compatible shapes.
        None if x, y or theta is not of the expected type.
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
    return (x * (y_hat - y)).sum(axis=0).reshape((2, 1)) / m
