import numpy as np
from numpy import ndarray


def typechecker(fun):
    def reshape_(*arg):
        return tuple(x.reshape((-1, 1)) if x.ndim == 1 else x for x in arg)

    def wrapper(x, theta):
        try:
            if isinstance(x, ndarray) and isinstance(theta, ndarray):
                x, theta = reshape_(x, theta)
                if (x.size and x.ndim == 2 and theta.size and theta.ndim == 2
                        and theta.shape == (x.shape[1] + 1, 1)):
                    return fun(x, theta)
        except Exception as e:
            print(e)
    return wrapper


def sigmoid_(x: ndarray) -> ndarray | None:
    return 1 / (1 + np.exp(-x))


@typechecker
def logistic_predict_(x: ndarray, theta: ndarray) -> ndarray | None:
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * n.
        theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
    Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
        None if x or theta are empty numpy.ndarray.
        None if x or theta dimensions are not appropriate.
    Raises:
        This function should not raise any Exception.
    """
    x1 = np.hstack((np.ones((x.shape[0], 1)), x))
    return sigmoid_(np.dot(x1, theta))


if __name__ == "__main__":
    # Example 1
    x = np.array([4]).reshape((-1, 1))
    theta = np.array([[2], [0.5]])
    print(logistic_predict_(x, theta))

    # Example 2
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    print(logistic_predict_(x2, theta2))

    # Example 3
    x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    print(logistic_predict_(x3, theta3))
