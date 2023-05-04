import numpy as np
from numpy import ndarray
from random import randint


def typechecker(fun):
    def reshape_(*arg):
        return tuple(x.reshape((-1, 1)) if x.ndim == 1 else x for x in arg)

    def wrapper(x, y, proportion):
        try:
            if (isinstance(x, ndarray) and isinstance(y, ndarray) and
                    isinstance(proportion, (int, float))):
                x, y = reshape_(x, y)
                if (x.size and x.ndim == 2 and
                        y.shape == (x.shape[0], 1) and 0 <= proportion <= 1):
                    return fun(x, y, proportion)
        except Exception as e:
            print(e)
    return wrapper


@typechecker
def data_spliter(x: ndarray, y: ndarray, proportion: float):
    """Shuffles and splits the dataset (given by x and y) into a training and a test set,
        while respecting the given proportion of examples to be kept in the training set.
    Args:
        x: has to be an numpy.array, a matrix of dimension m * n.
        y: has to be an numpy.array, a vector of dimension m * 1.
        proportion: has to be a float, the proportion of the dataset that will be assigned to the
        training set.
    Return:
        (x_train, x_test, y_train, y_test) as a tuple of numpy.array
        None if x or y is an empty numpy.array.
        None if x and y do not share compatible dimensions.
        None if x, y or proportion is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    m = x.shape[0]
    x, y = np.copy(x), np.copy(y)
    for i in range(m):
        j = randint(0, i)
        x[[i, j]] = x[[j, i]]
        y[[i, j]] = y[[j, i]]
    index = int(m * proportion)
    return x[:index], x[index:], y[:index], y[index:]


if __name__ == "__main__":

    x1 = np.array([1, 42, 300, 10, 59]).reshape((-1, 1))
    y = np.array([0, 1, 0, 1, 0]).reshape((-1, 1))
    # Example 1:
    print(data_spliter(x1, y, 0.8))

    # Example 2:
    print(data_spliter(x1, y, 0.5))

    x2 = np.array([[1, 42],
                   [300, 10],
                   [59, 1],
                   [300, 59],
                   [10, 42]])
    y = np.array([0, 1, 0, 1, 0]).reshape((-1, 1))
    # Example 3:
    print(data_spliter(x2, y, 0.8))

    # Example 4:
    print(data_spliter(x2, y, 0.5))
