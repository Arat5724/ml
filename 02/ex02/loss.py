import numpy as np
from numpy import ndarray


def typechecker(fun):
    def reshape_(*arg):
        return tuple(x.reshape((-1, 1)) if x.ndim == 1 else x for x in arg)

    def wrapper(y, y_hat):
        try:
            if isinstance(y, ndarray) and isinstance(y_hat, ndarray):
                y, y_hat = reshape_(y, y_hat)
                if y.size and y.ndim == 2 and y.shape == y_hat.shape:
                    return fun(y, y_hat)
        except Exception as e:
            print(e)
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
