import numpy as np
from numpy import ndarray


def add_polynomial_features(x: ndarray, power: int):
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
    # power must greater than or equal to 0
    x = x.reshape(-1)
    (m,) = x.shape
    v = np.empty((power, m), dtype=x.dtype)
    v[0] = x
    for i in range(1, power):
        v[i] = v[i - 1] * x
    return v.T
