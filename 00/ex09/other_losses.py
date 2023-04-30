import numpy as np
from numpy import ndarray
from math import sqrt


def typechecker(fun):
    def reshape_(*arg):
        return tuple(x.reshape((-1, 1)) if x.ndim == 1 else x for x in arg)

    def wrapper(y, y_hat):
        try:
            if isinstance(y, ndarray) and isinstance(y_hat, ndarray):
                y, y_hat = reshape_(y, y_hat)
                if y.size and y.ndim == 2 and y.shape == y_hat.shape:
                    return fun(y, y_hat)
        except Exception as e:
            print(e)
    return wrapper


@typechecker
def mse_(y: ndarray, y_hat: ndarray) -> float | None:
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
    res = (y_hat - y).reshape(-1)
    return np.dot(res, res) / y.shape[0]


@typechecker
def rmse_(y: ndarray, y_hat: ndarray) -> float | None:
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


@typechecker
def mae_(y: ndarray, y_hat: ndarray) -> float | None:
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
    return np.absolute(y_hat - y).sum() / y.shape[0]


@typechecker
def r2score_(y: ndarray, y_hat: ndarray) -> float | None:
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
    tmp2 = (y - y.mean()).reshape(-1)
    return 1 - np.dot(tmp1, tmp1) / np.dot(tmp2, tmp2)


if __name__ == "__main__":
    x = np.array([0, 15, -9, 7, 12, 3, -21])
    y = np.array([2, 14, -13, 5, 12, 4, -19])

    print(mse_(x, y))
    print(rmse_(x, y))
    print(mae_(x, y))
    print(r2score_(x, y))
