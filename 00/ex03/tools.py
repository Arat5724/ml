import numpy as np
from numpy import ndarray


def typechecker(fun):
    def wrapper(x):
        try:
            if isinstance(x, ndarray):
                x = x.reshape((-1, 1)) if x.ndim == 1 else x
                if x.size and x.ndim == 2:
                    return fun(x)
        except Exception as e:
            print(e)
    return wrapper


@typechecker
def add_intercept(x: ndarray) -> ndarray | None:
    """
    Adds a column of 1's to the non-empty numpy.array x.
    Args:
        x: has to be a numpy.array of dimension m * n.
    Returns:
        X, a numpy.array of dimension m * (n + 1).
        None if x is not a numpy.array.
        None if x is an empty numpy.array.
    Raises:
        This function should not raise any Exception.
    """
    return np.hstack((np.ones((x.shape[0], 1)), x))


if __name__ == "__main__":
    x = np.arange(1, 6)
    print(add_intercept(x))
    y = np.arange(1, 10).reshape((3, 3))
    print(add_intercept(y))
