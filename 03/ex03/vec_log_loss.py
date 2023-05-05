import numpy as np
from numpy import ndarray


def typechecker(fun):
    def reshape_(*arg):
        return tuple(x.reshape((-1, 1)) if x.ndim == 1 else x for x in arg)

    def wrapper(y, y_hat, eps=1e-15):
        try:
            if (isinstance(y, ndarray) and isinstance(y_hat, ndarray) and
                    isinstance(eps, float)):
                y, y_hat = reshape_(y, y_hat)
                if (y.size and y.ndim == 2 and y.shape == y_hat.shape):
                    return fun(y, y_hat, eps)
        except Exception as e:
            print(e)
    return wrapper


def sigmoid_(x: ndarray) -> ndarray | None:
    return 1 / (1 + np.exp(-x))


def logistic_predict_(x: ndarray, theta: ndarray) -> ndarray | None:
    x1 = np.hstack((np.ones((x.shape[0], 1)), x))
    return sigmoid_(np.dot(x1, theta))


@typechecker
def vec_log_loss_(y: ndarray, y_hat: ndarray, eps: float = 1e-15) -> float | None:
    """
    Compute the logistic loss value.
    Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        eps: epsilon (default=1e-15)
    Returns:
        The logistic loss value as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    m, n = y.shape
    y, y_hat = y.reshape(-1), y_hat.reshape(-1)
    return (np.dot(y, np.log(y_hat + eps)) + np.dot(1 - y, np.log(1 - y_hat + eps))) / -m


if __name__ == "__main__":
    # Example 1:
    y1 = np.array([1]).reshape((-1, 1))
    x1 = np.array([4]).reshape((-1, 1))
    theta1 = np.array([[2], [0.5]])
    y_hat1 = logistic_predict_(x1, theta1)
    print(vec_log_loss_(y1, y_hat1))

    # Example 2:
    y2 = np.array([[1], [0], [1], [0], [1]])
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    y_hat2 = logistic_predict_(x2, theta2)
    print(vec_log_loss_(y2, y_hat2))
    # Output:

    # Example 3:
    y3 = np.array([[0], [1], [1]])
    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    y_hat3 = logistic_predict_(x3, theta3)
    print(vec_log_loss_(y3, y_hat3))
