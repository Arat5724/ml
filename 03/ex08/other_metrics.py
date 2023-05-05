import numpy as np
from numpy import ndarray


def typechecker1(fun):
    def wrapper(y, y_hat):
        try:
            if (isinstance(y, ndarray) and isinstance(y_hat, ndarray) and
                    y.shape == y_hat.shape):
                return fun(y, y_hat)
        except Exception as e:
            print(e)
    return wrapper


def typechecker2(fun):
    def wrapper(y, y_hat, pos_label=1):
        try:
            if (isinstance(y, ndarray) and isinstance(y_hat, ndarray) and
                    y.shape == y_hat.shape and isinstance(pos_label, (str, int))):
                return fun(y, y_hat, pos_label)
        except Exception as e:
            print(e)
    return wrapper


@typechecker1
def accuracy_score_(y: ndarray, y_hat: ndarray) -> float | None:
    """
    Compute the accuracy score.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
    Returns:
        The accuracy score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    return (y == y_hat).sum() / y.size


@typechecker2
def precision_score_(y: ndarray, y_hat: ndarray, pos_label: str | int = 1) -> float | None:
    """
    Compute the precision score.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
        pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
        The precision score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    y = y[(y_hat == pos_label)]
    return (y == pos_label).sum() / y.size


@typechecker2
def recall_score_(y: ndarray, y_hat: ndarray, pos_label: str | int = 1) -> float | None:
    """
    Compute the recall score.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
        pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
        The recall score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    y_hat = y_hat[(y == pos_label)]
    return (y_hat == pos_label).sum() / y_hat.size


@typechecker2
def f1_score_(y: ndarray, y_hat: ndarray, pos_label: str | int = 1) -> float | None:
    """
    Compute the f1 score.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
        pos_label: str or int, the class on which to report the precision_score (default=1)
    Returns:
        The f1 score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    p = precision_score_(y, y_hat, pos_label)
    r = recall_score_(y, y_hat, pos_label)
    return 2 * p * r / (p + r)


if __name__ == "__main__":
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # Example 1:
    y_hat = np.array([1, 1, 0, 1, 0, 0, 1, 1]).reshape((-1, 1))
    y = np.array([1, 0, 0, 1, 0, 1, 0, 0]).reshape((-1, 1))

    # Accuracy
    # your implementation
    print(accuracy_score_(y, y_hat))
    # sklearn implementation
    print(accuracy_score(y, y_hat))

    # Precision
    # your implementation
    print(precision_score_(y, y_hat))
    # sklearn implementation
    print(precision_score(y, y_hat))

    # Recall
    # your implementation
    print(recall_score_(y, y_hat))
    # sklearn implementation
    print(recall_score(y, y_hat))

    # F1-score
    # your implementation
    print(f1_score_(y, y_hat))
    # sklearn implementation
    print(f1_score(y, y_hat))

    # Example 2:
    y_hat = np.array(['norminet', 'dog', 'norminet',
                     'norminet', 'dog', 'dog', 'dog', 'dog'])
    y = np.array(['dog', 'dog', 'norminet', 'norminet',
                 'dog', 'norminet', 'dog', 'norminet'])
    # Accuracy
    # your implementation
    print(accuracy_score_(y, y_hat))
    # sklearn implementation
    print(accuracy_score(y, y_hat))

    # Precision
    # your implementation
    print(precision_score_(y, y_hat, pos_label='dog'))
    # sklearn implementation
    print(precision_score(y, y_hat, pos_label='dog'))

    # Recall
    # your implementation
    print(recall_score_(y, y_hat, pos_label='dog'))
    # sklearn implementation
    print(recall_score(y, y_hat, pos_label='dog'))

    # F1-score
    # your implementation
    print(f1_score_(y, y_hat, pos_label='dog'))
    # sklearn implementation
    print(f1_score(y, y_hat, pos_label='dog'))

    # Example 3:
    y_hat = np.array(['norminet', 'dog', 'norminet',
                     'norminet', 'dog', 'dog', 'dog', 'dog'])
    y = np.array(['dog', 'dog', 'norminet', 'norminet',
                 'dog', 'norminet', 'dog', 'norminet'])
    # Precision
    # your implementation
    print(precision_score_(y, y_hat, pos_label='norminet'))
    # sklearn implementation
    print(precision_score(y, y_hat, pos_label='norminet'))

    # Recall
    # your implementation
    print(recall_score_(y, y_hat, pos_label='norminet'))
    # sklearn implementation
    print(recall_score(y, y_hat, pos_label='norminet'))

    # F1-score
    # your implementation
    print(f1_score_(y, y_hat, pos_label='norminet'))
    # sklearn implementation
    print(f1_score(y, y_hat, pos_label='norminet'))
