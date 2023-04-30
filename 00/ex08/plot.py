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


def loss_(y: ndarray, y_hat: ndarray) -> float | None:
    res = (y_hat - y).reshape(-1)
    return np.dot(res, res) / 2 / y.shape[0]


@typechecker
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

    y_hat = predict_(x, theta)
    cost = loss_(y, y_hat)
    ax = ax = plt.figure(num="plot").add_subplot()
    ax.set_title(f'Cost: {cost}')
    ax.scatter(x, y)
    ax.plot(x, y_hat, c="orange")
    for x0, y0, y_hat0 in zip(x, y, y_hat):
        ax.plot([x0, x0], [y0, y_hat0], ls='--')
    plt.show()


if __name__ == "__main__":
    x = np.arange(1, 6)
    y = np.array([11.52434424, 10.62589482,
                 13.14755699, 18.60682298, 14.14329568])
    theta1 = np.array([18, -1])
    plot_with_loss(x, y, theta1)

    theta2 = np.array([14, 0])
    plot_with_loss(x, y, theta2)

    theta3 = np.array([12, 0.8])
    plot_with_loss(x, y, theta3)
