import numpy as np
from numpy import ndarray


def typechecker(fun):
    def wrapper(x):
        try:
            if isinstance(x, ndarray) and x.size:
                return fun(x)
        except Exception as e:
            print(e)
    return wrapper


@typechecker
def minmax(x: ndarray) -> ndarray | None:
    """
    Computes the normalized version of a non-empty numpy.ndarray using the min-max standardization.
    Args:
        x: has to be an numpy.ndarray, a vector.
    Returns:
        x' as a numpy.ndarray.
        None if x is a non-empty numpy.ndarray or not a numpy.ndarray.
    Raises:
        This function shouldn't raise any Exception.
    """
    if (xmax := x.max()) - (xmin := x.min()) == 0:
        return np.zeros_like(x)
    return (x - xmin)/(xmax - xmin)


if __name__ == "__main__":
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((-1, 1))
    print(minmax(X))
    scaler.fit(X)
    print(scaler.transform(X))

    Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    print(minmax(Y))
    scaler.fit(Y)
    print(scaler.transform(Y))
