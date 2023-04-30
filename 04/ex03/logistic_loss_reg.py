import numpy as np
from numpy import ndarray


def l2(theta):
    theta = theta.reshape(-1)
    theta[0] = 0.0
    return np.dot(theta, theta)


def reg_log_loss_(y, y_hat, theta, lambda_):
    """Computes the regularized loss of a logistic regression model from two non-empty numpy.ndarray, without any for loop
    Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
        lambda_: has to be a float.
    Returns:
        The regularized loss as a float.
        None if y, y_hat, or theta is empty numpy.ndarray.
        None if y and y_hat do not share the same shapes.
    Raises:
        This function should not raise any Exception.
    """
    m, n = y.shape

    y, y_hat = y.reshape(-1), y_hat.reshape(-1)
    return (np.dot(y, np.log(y_hat + 1e-15)) + np.dot(1 - y, np.log(1 - y_hat + 1e-15))) / -m + \
        + lambda_ * l2(theta) / 2 / m
    # return (np.dot(y, np.log(y_hat + eps)) + np.dot(1 - y, np.log(1 - y_hat + eps))) / -m
