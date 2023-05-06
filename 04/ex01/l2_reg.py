import numpy as np
from numpy import ndarray


def typechecker(fun):
    def wrapper(theta):
        try:
            if isinstance(theta, ndarray) and theta.size:
                theta = theta.reshape((-1, 1)) if theta.ndim == 1 else theta
                if theta.ndim == 2:
                    return fun(theta)
        except Exception as e:
            print(e)
    return wrapper


@typechecker
def iterative_l2(theta: ndarray) -> float | None:
    """Computes the L2 regularization of a non-empty numpy.ndarray, with a for-loop.
    Args:
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
    Returns:
        The L2 regularization as a float.
        None if theta in an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    res = 0.0
    it = iter(theta.reshape(-1))
    next(it)
    for i in it:
        res += i * i
    return res


@typechecker
def l2(theta: ndarray) -> float | None:
    """Computes the L2 regularization of a non-empty numpy.ndarray, without any for-loop.
    Args:
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
    Returns:
        The L2 regularization as a float.
        None if theta in an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    theta1 = theta[1:].reshape(-1)
    return np.dot(theta1, theta1) / 1


if __name__ == "__main__":
    x = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    # Example 1:
    print(iterative_l2(x))

    # Example 2:
    print(l2(x))

    y = np.array([3, 0.5, -6]).reshape((-1, 1))
    # Example 3:
    print(iterative_l2(y))

    # Example 4:
    print(l2(y))
