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
                if (x.size and theta.size and x.ndim == 2 and theta.ndim == 2 and
                        x.shape[1] == 1 and theta.shape == (2, 1) and x.shape == y.shape):
                    return fun(x, y, theta)
        except Exception as e:
            print(e)
    return wrapper


@typechecker
def simple_gradient(x: ndarray, y: ndarray, theta: ndarray) -> ndarray | None:
    """
    Computes a gradient vector from three non-empty numpy.array, with a for-loop.
        The three arrays must have compatible shapes.
    Args:
        x: has to be an numpy.array, a vector of shape m * 1.
        y: has to be an numpy.array, a vector of shape m * 1.
        theta: has to be an numpy.array, a 2 * 1 vector.
    Return:
        The gradient as a numpy.array, a vector of shape 2 * 1.
        None if x, y, or theta are empty numpy.array.
        None if x, y and theta do not have compatible shapes.
        None if x, y or theta is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    j0, j1 = 0, 0
    for xi, yi in zip(x, y):
        tmp = theta[0] + theta[1] * xi - yi
        j0 += tmp
        j1 += tmp * xi
    return np.array([j0, j1]) / y.shape[0]


if __name__ == "__main__":
    x = np.array([12.4956442, 21.5007972, 31.5527382,
                 48.9145838, 57.5088733]).reshape((-1, 1))
    y = np.array([37.4013816, 36.1473236, 45.7655287,
                 46.6793434, 59.5585554]).reshape((-1, 1))

    theta1 = np.array([2, 0.7]).reshape((-1, 1))
    print(simple_gradient(x, y, theta1))

    theta2 = np.array([1, -0.4]).reshape((-1, 1))
    print(simple_gradient(x, y, theta2))
