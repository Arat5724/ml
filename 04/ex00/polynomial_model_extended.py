import numpy as np
from numpy import ndarray


def typechecker(fun):
    def wrapper(x, power):
        try:
            if isinstance(x, ndarray) and isinstance(power, int):
                if (x.size and x.ndim == 2 and power > 0):
                    return fun(x, power)
        except Exception as e:
            print(e)
    return wrapper


@typechecker
def add_polynomial_features(x: ndarray, power: int) -> ndarray | None:
    """Add polynomial features to vector x by raising its values up to the power given in argument.
    Args:
        x: has to be an numpy.array, a vector of dimension m * n.
        power: has to be an int, the power up to which the components of vector x are going to be raised.
    Return:
        The matrix of polynomial features as a numpy.array, of dimension m * (np),
        containing the polynomial feature values for all training examples.
        None if x is an empty numpy.array.
        None if x or power is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    (m, n) = x.shape
    v = np.empty((n * power, m), dtype=x.dtype)
    xT = x.T
    v[:n] = xT
    for i in range(1, power):
        v[n * i: n * (i + 1)] = v[n * (i - 1): n * i] * xT
    return v.T


if __name__ == "__main__":
    x = np.arange(1, 11).reshape(5, 2)

    # Example 1:
    print(add_polynomial_features(x, 3))

    # Example 2:
    print(add_polynomial_features(x, 4))
