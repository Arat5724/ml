import numpy as np
from numpy import ndarray
from random import randint


def data_spliter(x, y, proportion):
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
    m, _ = x.shape
    x, y = np.copy(x), np.copy(y)
    for i in range(m):
        j = randint(0, i)
        x[[i, j]] = x[[j, i]]
        y[[i, j]] = y[[j, i]]
    m0 = int(m * proportion)
    return x[:m0], x[m0:], y[:m0], y[m0:]
