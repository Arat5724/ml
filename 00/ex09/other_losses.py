import numpy as np
from numpy import ndarray
from math import sqrt


def set_dimension(fun):
    def wrapper(y: ndarray, y_hat: ndarray) -> float:
        if y.ndim == 1:
            y = y.reshape((*y.shape, 1))
        if y_hat.ndim == 1:
            y_hat = y_hat.reshape((*y_hat.shape, 1))
        return fun(y, y_hat)
    return wrapper


@set_dimension
def mse_(y: ndarray, y_hat: ndarray) -> float:
    """
    Description:
        Calculate the MSE between the predicted output and the real output.
    Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
        mse: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    (m, _) = y.shape
    res = (y_hat - y).reshape(-1)
    return np.dot(res, res) / m


@set_dimension
def rmse_(y: ndarray, y_hat: ndarray) -> float:
    """
    Description:
        Calculate the RMSE between the predicted output and the real output.
    Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
        rmse: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    return sqrt(mse_(y, y_hat))


@set_dimension
def mae_(y: ndarray, y_hat: ndarray) -> float:
    """
    Description:
        Calculate the MAE between the predicted output and the real output.
    Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
        mae: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    (m, _) = y.shape
    return (y_hat - y).reshape(-1).absolute().sum() / m


@set_dimension
def r2score_(y: ndarray, y_hat: ndarray) -> float:
    """
    Description:
        Calculate the R2score between the predicted output and the output.
    Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
        r2score: has to be a float.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exceptions.
    """
    tmp1 = (y_hat - y).reshape(-1)
    tmp2 = (y - np.full_like(y, y.mean())).reshape(-1)
    print(tmp2)
    return 1 - np.dot(tmp1, tmp1) / np.dot(tmp2, tmp2)
