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

    x = add_intercept(x)
    if y.ndim == 1:
        y = y.reshape((*y.shape, 1))
    if theta.ndim == 1:
        theta = theta.reshape((*theta.shape, 1))
    m, _ = y.shape
    y_hat = np.dot(x, theta)
    return np.dot(x.T, y_hat - y).reshape((2, 1)) / m


def fit_(x: ndarray, y: ndarray, theta: ndarray, alpha: float, max_iter: int) -> ndarray:
    """
    Description:
        Fits the model to the training dataset contained in x and y.
    Args:
        x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the gradient descent
    Returns:
        new_theta: numpy.ndarray, a vector of dimension 2 * 1.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exception.
    """
    for _ in range(max_iter):
        new_theta = theta - alpha * gradient(x, y, theta)
        if np.array_equal(new_theta, theta):
            break
        theta = new_theta
    return theta
