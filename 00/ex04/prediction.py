import numpy as np
from numpy import ndarray


def typechecker(fun):
    def reshape_(*arg):
        return tuple(x.reshape((-1, 1)) if x.ndim == 1 else x for x in arg)

    def wrapper(x, theta):
        try:
            if isinstance(x, ndarray) and isinstance(theta, ndarray):
                x, theta = reshape_(x, theta)
                if (x.size and theta.size and
                    x.ndim == 2 and theta.ndim == 2 and
                        x.shape[1] == 1 and theta.shape == (2, 1)):
                    return fun(x, theta)
        except Exception as e:
            print(e)
    return wrapper


def add_intercept(x: ndarray) -> ndarray | None:
    return np.hstack((np.ones((x.shape[0], 1)), x))


@typechecker
def predict_(x: ndarray, theta: ndarray) -> ndarray | None:
    """
    Computes the vector of prediction y_hat from two non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a vector of dimension m * 1.
        theta: has to be an numpy.array, a vector of dimension 2 * 1.
    Returns:
        y_hat as a numpy.array, a vector of dimension m * 1.
        None if x and/or theta are not numpy.array.
        None if x or theta are empty numpy.array.
        None if x or theta dimensions are not appropriate.
    Raises:
        This function should not raise any Exceptions.
    """
    x1 = add_intercept(x)
    return np.dot(x1, theta)


if __name__ == "__main__":
    x = np.arange(1, 6)
    # Example 1:
    theta1 = np.array([5, 0])
    print(predict_(x, theta1))

    # Example 2:
    theta2 = np.array([0, 1])
    print(predict_(x, theta2))

    # Example 3:
    theta3 = np.array([5, 3])
    print(predict_(x, theta3))

    # Example 4:
    theta4 = np.array([-3, 1])
    print(predict_(x, theta4))
