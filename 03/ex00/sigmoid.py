import numpy as np
from numpy import ndarray


def typechecker(fun):
    def wrapper(x):
        try:
            if isinstance(x, ndarray):
                x = x.reshape((-1, 1)) if x.ndim == 1 else x
                if (x.size and x.ndim == 2 and x.shape[1] == 1):
                    return fun(x)
        except Exception as e:
            print(e)
    return wrapper


@typechecker
def sigmoid_(x: ndarray) -> ndarray | None:
    """
    Compute the sigmoid of a vector.
    Args:
        x: has to be a numpy.ndarray of shape (m, 1).
    Returns:
        The sigmoid value as a numpy.ndarray of shape (m, 1).
        None if x is an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    x = np.array([[-4]])
    print(sigmoid_(x))

    # Example 2:
    x = np.array([[2]])
    print(sigmoid_(x))

    # Example 3:
    x = np.array([[-4], [2], [0]])
    print(sigmoid_(x))
