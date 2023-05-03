import numpy as np
from numpy import ndarray


def typechecker(fun):
    def wrapper(x, power):
        try:
            if isinstance(x, ndarray) and isinstance(power, int):
                if (x.size and x.ndim == 2 and x.shape[1] == 1 and power > 0):
                    return fun(x, power)
        except Exception as e:
            print(e)
    return wrapper


@typechecker
def add_polynomial_features(x: ndarray, power: int) -> ndarray | None:
    """Add polynomial features to vector x by raising its values up to the power given in argument.
    Args:
        x: has to be an numpy.array, a vector of dimension m * 1.
        power: has to be an int, the power up to which the components of vector x are going to be raised.
    Return:
        The matrix of polynomial features as a numpy.array, of dimension m * n,
        containing the polynomial feature values for all training examples.
        None if x is an empty numpy.array.
        None if x or power is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    x = x.reshape(-1)
    v = np.empty((power, x.shape[0]), dtype=x.dtype)
    v[0] = x
    for i in range(1, power):
        v[i] = v[i - 1] * x
    return v.T


if __name__ == "__main__":
    x = np.arange(1, 6).reshape(-1, 1)
    # Example 0:
    print(add_polynomial_features(x, 3))
    # Example 1:
    print(add_polynomial_features(x, 6))
