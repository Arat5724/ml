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
def loss_(y: ndarray, y_hat: ndarray) -> float | None:
    """Computes the mean squared error of two non-empty numpy.array, without any for loop.
    The two arrays must have the same dimensions.
    Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
    Return:
        The mean squared error of the two vectors as a float.
        None if y or y_hat are empty numpy.array.
        None if y and y_hat does not share the same dimensions.
        None if y or y_hat is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    res = (y_hat - y).reshape(-1)
    return np.dot(res, res) / 2 / y.shape[0]


if __name__ == "__main__":

    X = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
    Y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])

    print(loss_(X, Y))
    print(loss_(X, X))
