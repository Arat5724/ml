import numpy as np
from numpy import ndarray


def typechecker(fun):
    def reshape_(*arg):
        return tuple(x.reshape((-1, 1)) if x.ndim == 1 else x for x in arg)

    def wrapper(x, y, theta):
        try:
            if (isinstance(x, ndarray) and isinstance(y, ndarray) and
                    isinstance(theta, ndarray)):
                x, y, theta = reshape_(x, y, theta)
                if (x.size and x.ndim == 2 and
                        y.shape == (x.shape[0], 1) and theta.shape == (x.shape[1] + 1, 1)):
                    return fun(x, y, theta)
        except Exception as e:
            print(e)
    return wrapper


def add_intercept(x: ndarray) -> ndarray | None:
    return np.hstack((np.ones((x.shape[0], 1)), x))


@typechecker
def gradient(x: ndarray, y: ndarray, theta: ndarray) -> ndarray | None:
    """Computes a gradient vector from three non-empty numpy.array, without any for-loop.
        The three arrays must have the compatible dimensions.
    Args:
        x: has to be an numpy.array, a matrix of dimension m * n.
        y: has to be an numpy.array, a vector of dimension m * 1.
        theta: has to be an numpy.array, a vector (n + 1) * 1.
    Return:
        The gradient as a numpy.array, a vector of dimensions n * 1,
        containg the result of the formula for all j.
        None if x, y, or theta are empty numpy.array.
        None if x, y and theta do not have compatible dimensions.
        None if x, y or theta is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    x1 = add_intercept(x)
    y_hat = np.dot(x1, theta)
    return np.dot(x1.T, y_hat - y) / x.shape[0]


if __name__ == "__main__":
    x = np.array([
        [-6, -7, -9],
        [13, -2, 14],
        [-7, 14, -1],
        [-8, -4, 6],
        [-5, -9, 6],
        [1, -5, 11],
        [9, -11, 8]])
    y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))

    theta1 = np.array([0, 3, 0.5, -6]).reshape((-1, 1))
    # Example :
    print(gradient(x, y, theta1))

    # Example :
    theta2 = np.array([0, 0, 0, 0]).reshape((-1, 1))
    print(gradient(x, y, theta2))
