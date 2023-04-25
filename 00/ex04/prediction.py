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


def predict_(x: ndarray, theta: ndarray) -> ndarray:
    """
    Computes the vector of prediction y_hat from two non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a vector of dimension m * 1.
        theta: has to be an numpy.array, a vector of dimension 2 * 1.
    Returns:
        y_hat as a numpy.array, a vector of dimension m * 1.
        None if x and/or theta are not numpy.array.
        None if x or theta are empty numpy.array.
        None if x or theta dimensions are not appropriate.
    Raises:
        This function should not raise any Exceptions.
    """
    try:
        x = add_intercept(x)
        if x is None:
            return None
        if theta.ndim == 1:
            theta = theta.reshape((*theta.shape, 1))
        t, k = theta.shape
        if t == 2 and k == 1:
            return x.dot(theta)
    except Exception as e:
        print(e)
