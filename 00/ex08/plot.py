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


def loss_(y: ndarray, y_hat: ndarray) -> float:
    (m, _) = y.shape
    res = (y_hat - y).reshape(-1)
    return np.dot(res, res) / 2 / m


def plot_with_loss(x: ndarray, y: ndarray, theta: ndarray) -> None:
    """
    Plot the data and prediction line from three non-empty numpy.ndarray.
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * 1.
        y: has to be an numpy.ndarray, a vector of dimension m * 1.
        theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
        Nothing.
    Raises:
        This function should not raise any Exception.
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
    y_hat = predict_(x, theta)
    cost = loss_(y, y_hat)
    fig, ax = plt.subplots(num="plot")
    ax.set_title(f'Cost: {cost}')
    ax.scatter(x, y)
    ax.plot(x, y_hat, c="orange")
    for x0, y0, y_hat0 in zip(x, y, y_hat):
        ax.plot([x0, x0], [y0, y_hat0], ls='--')
    plt.show()
