import numpy as np
from numpy import ndarray


def simple_predict(x: ndarray, theta: ndarray) -> ndarray:
    """
    Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * 1.
        theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
        None if x or theta are empty numpy.ndarray.
        None if x or theta dimensions are not appropriate.
    Raises:
        This function should not raise any Exception.
    """

    try:
        if x.ndim == 1:
            x = x.reshape((*x.shape, 1))
        m, n = x.shape
        if theta.ndim == 1:
            theta = theta.reshape((*theta.shape, 1))
        t, k = theta.shape
        if m > 0 and n == 1 and t == 2 and k == 1:
            return theta[0][0] + theta[1][0] * x
    except Exception as e:
        print(e)
