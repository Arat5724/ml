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


def loss_elem_(y: ndarray, y_hat: ndarray) -> ndarray:
    """
    Description:
        Calculates all the elements (y_pred - y)^2 of the loss function.
    Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
    Returns:
        J_elem: numpy.array, a vector of dimension (number of the training examples,1).
        None if there is a dimension matching problem between X, Y or theta.
        None if any argument is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    res = y_hat - y
    return res * res


def loss_(y: ndarray, y_hat: ndarray) -> float:
    """
    Description:
        Calculates the value of loss function.
    Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
    Returns:
        J_value : has to be a float.
        None if there is a dimension matching problem between X, Y or theta.
        None if any argument is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    if y.shape == y_hat.shape:
        m, _ = y.shape
        loss = loss_elem_(y, y_hat)
        return loss.sum() / 2 / m
