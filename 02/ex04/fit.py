import numpy as np
from numpy import ndarray


def predict_(x: ndarray, theta: ndarray) -> ndarray:
    m, _ = x.shape
    return np.dot(np.hstack((np.ones((m, 1)), x)), theta)


def gradient(x: ndarray, y: ndarray, theta: ndarray) -> ndarray:
    m, _ = x.shape
    x = np.hstack((np.ones((m, 1)), x))
    y_hat = np.dot(x, theta)
    return np.dot(x.T, y_hat - y) / m


def fit_(x: ndarray, y: ndarray, theta: ndarray, alpha: float, max_iter: int) -> ndarray:
    """
    Description:
        Fits the model to the training dataset contained in x and y.
    Args:
         x: has to be a numpy.array, a matrix of dimension m * n:
                        (number of training examples, number of features).
         y: has to be a numpy.array, a vector of dimension m * 1:
                        (number of training examples, 1).
         theta: has to be a numpy.array, a vector of dimension (n + 1) * 1:
                        (number of features + 1, 1).
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the gradient descent
    Return:
        new_theta: numpy.array, a vector of dimension (number of features + 1, 1).
        None if there is a matching dimension problem.
        None if x, y, theta, alpha or max_iter is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    for _ in range(max_iter):
        new_theta = theta - alpha * gradient(x, y, theta)
        if np.array_equal(new_theta, theta):
            break
        theta = new_theta
    return theta
