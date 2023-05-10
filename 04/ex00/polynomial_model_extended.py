import numpy as np
from numpy import ndarray


def typechecker(fun):
    def wrapper(*args, **kwargs):
        from inspect import signature
        bound_args = signature(fun).bind(*args, **kwargs).arguments
        for param, value in bound_args.items():
            if param in fun.__annotations__ and not isinstance(value, fun.__annotations__[param]):
                return None
            if isinstance(value, ndarray) and (value.size == 0 or value.ndim != 2):
                return None
        theta, x, y, y_hat = [bound_args[ele] if ele in bound_args else None
                              for ele in ['theta', 'x', 'y', 'y_hat']]
        if ((theta is not None and ((x is not None and theta.shape[0] != x.shape[1] + 1) or theta.shape[1] != 1)) or
            (y is not None and ((x is not None and y.shape[0] != x.shape[0]) or y.shape[1] != 1)) or
            (y_hat is not None and ((y is not None and y_hat.shape != y.shape) or
                                    (y is None and ((x is not None and y_hat.shape[0] != x.shape[0]) or y_hat.shape[1] != 1))))):
            return None
        try:
            return fun(*args, **kwargs)
        except:
            return None
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
    if power < 1:
        return None
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
