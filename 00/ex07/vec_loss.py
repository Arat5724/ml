import numpy as np
from numpy import ndarray

def loss_(y: ndarray, y_hat: ndarray) -> float:
    """
    Computes the half mean squared error of two non-empty numpy.array, without any for loop.
        The two arrays must have the same dimensions.
    Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
    Returns:
        The half mean squared error of the two vectors as a float.
        None if y or y_hat are empty numpy.array.
        None if y and y_hat does not share the same dimensions.
    Raises:
        This function should not raise any Exceptions.
    """
    (m, _) = y.shape
    res = (y_hat - y).reshape(-1)
    return np.dot(res, res) / 2 / m
