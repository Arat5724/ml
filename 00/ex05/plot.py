import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt


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


def add_intercept(x: ndarray) -> ndarray | None:
    return np.hstack((np.ones((x.shape[0], 1)), x))


def predict_(x: ndarray, theta: ndarray) -> ndarray | None:
    x1 = add_intercept(x)
    return np.dot(x1, theta)


@typechecker
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
    ax = plt.figure(num="plot").add_subplot()
    ax.scatter(x, y)
    y_hat = predict_(x, theta)
    ax.plot(x, y_hat, c="orange")
    plt.show()


if __name__ == "__main__":
    x = np.arange(1, 6)
    y = np.array([3.74013816, 3.61473236, 4.57655287, 4.66793434, 5.95585554])
    theta1 = np.array([[4.5], [-0.2]])
    plot(x, y, theta1)

    theta2 = np.array([[-1.5], [2]])
    plot(x, y, theta2)

    theta3 = np.array([[3], [0.3]])
    plot(x, y, theta3)
