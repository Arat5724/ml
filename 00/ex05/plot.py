import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt


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


def plot(x: ndarray, y: ndarray, theta: ndarray) -> None:
    """
    Plot the data and prediction line from three non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a vector of dimension m * 1.
        y: has to be an numpy.array, a vector of dimension m * 1.
        theta: has to be an numpy.array, a vector of dimension 2 * 1.
    Returns:
        Nothing.
    Raises:
        This function should not raise any Exceptions.
    """
    if x.ndim == 1:
        x = x.reshape((*x.shape, 1))
    if y.ndim == 1:
        y = y.reshape((*y.shape, 1))
    if x.shape != y.shape:
        return None
    if theta.ndim == 1:
        theta = theta.reshape((*theta.shape, 1))
    m, _ = x.shape
    t, k = theta.shape
    fig, ax = plt.subplots(num="plot")
    ax.scatter(x, y)
    ax.plot(x, predict_(x, theta), c="orange")
    plt.show()
