import numpy as np
from numpy import ndarray


def sigmoid_(x):
    return 1 / (1 + np.exp(-x))


def reg_logistic_grad(y, x, theta, lambda_):
    """Computes the regularized logistic gradient of three non-empty numpy.ndarray,
    with two for-loops. The three arrays must have compatible shapes.
    Args:
        y: has to be a numpy.ndarray, a vector of shape m * 1.
        x: has to be a numpy.ndarray, a matrix of dimesion m * n.
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
        lambda_: has to be a float.
    Returns:
        A numpy.ndarray, a vector of shape n * 1, containing the results of the formula for all j.
        None if y, x, or theta are empty numpy.ndarray.
        None if y, x or theta does not share compatibles shapes.
    Raises:
        This function should not raise any Exception.
    """
    m, n = x.shape
    x1 = np.hstack((np.ones((m, 1), dtype=x.dtype), x))
    theta1 = np.copy(theta)
    theta1[0] = 0.0
    y_hat = sigmoid_(np.dot(x1, theta))
    ysub = (y_hat - y).reshape(-1)
    nabla = np.zeros_like(theta, dtype=float)
    for j in range(n + 1):
        for i in range(m):
            nabla[j] += ysub[i] * x1[i][j]
        nabla[j] += lambda_ * theta1[j]
        nabla[j] /= m
    return nabla


def vec_reg_logistic_grad(y, x, theta, lambda_):
    """Computes the regularized logistic gradient of three non-empty numpy.ndarray,
    without any for-loop. The three arrays must have compatible shapes.
    Args:
        y: has to be a numpy.ndarray, a vector of shape m * 1.
        x: has to be a numpy.ndarray, a matrix of shape m * n.
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
        lambda_: has to be a float.
    Returns:
        A numpy.ndarray, a vector of shape n * 1, containing the results of the formula for all j.
        None if y, x, or theta are empty numpy.ndarray.
        None if y, x or theta does not share compatibles shapes.
    Raises:
        This function should not raise any Exception.
    """
    m, n = x.shape
    x1 = np.hstack((np.ones((m, 1), dtype=x.dtype), x))
    theta1 = np.copy(theta)
    theta1[0] = 0.0
    y_hat = sigmoid_(np.dot(x1, theta))
    return (np.dot(x1.T, y_hat - y) + lambda_ * theta1) / m
