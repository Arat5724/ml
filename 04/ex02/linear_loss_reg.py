import numpy as np
from numpy import ndarray


def reg_loss_(y, y_hat, theta, lambda_):
    """Computes the regularized loss of a linear regression model from two non-empty numpy.array, without any for loop.
    Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
        lambda_: has to be a float.
    Returns:
        The regularized loss as a float.
        None if y, y_hat, or theta are empty numpy.ndarray.
        None if y and y_hat do not share the same shapes.
    Raises:
        This function should not raise any Exception.
    """
    (m, _) = y.shape
    res, theta = (y_hat - y).reshape(-1), theta.reshape(-1)
    theta[0] = 0.0
    return (np.dot(res, res) + lambda_ * np.dot(theta, theta)) / 2 / m
