import numpy as np
from numpy import ndarray
from pandas import DataFrame


def typechecker(fun):
    def wrapper(y_true, y_hat, labels=None, df_option=False):
        try:
            if (isinstance(y_true, ndarray) and isinstance(y_hat, ndarray) and
                isinstance(labels, (list, type(None))) and isinstance(df_option, bool) and
                    y_true.shape == y_hat.shape):
                return fun(y_true, y_hat, labels, df_option)
        except Exception as e:
            print(e)
    return wrapper


@typechecker
def confusion_matrix_(y_true: ndarray, y_hat: ndarray,
                      labels: list | None = None, df_option: bool = False) -> ndarray | DataFrame | None:
    """
    Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
        y:a numpy.array for the correct labels
        y_hat:a numpy.array for the predicted labels
        labels: optional, a list of labels to index the matrix.
        This may be used to reorder or select a subset of labels. (default=None)
        df_option: optional, if set to True the function will return a pandas DataFrame
        instead of a numpy array. (default=False)
    Return:
        The confusion matrix as a numpy array or a pandas DataFrame according to df_option value.
        None if any error.
    Raises:
        This function should not raise any Exception.
    """
    y_true, y_hat = y_true.reshape(-1), y_hat.reshape(-1)
    if labels is None:
        labels = list(set(y_true.tolist() + y_hat.tolist()))
        labels.sort()
    true = np.vstack([y_true == label for label in labels], dtype=int)
    prediction = np.vstack([y_hat == label for label in labels], dtype=int)
    if df_option:
        return DataFrame(data=np.dot(true, prediction.T), index=labels, columns=labels)
    return np.dot(true, prediction.T)


if __name__ == "__main__":
    from sklearn.metrics import confusion_matrix

    y_hat = np.array([['norminet'], ['dog'], ['norminet'],
                     ['norminet'], ['dog'], ['bird']])
    y = np.array([['dog'], ['dog'], ['norminet'], [
                 'norminet'], ['dog'], ['norminet']])

    # Example 1:
    # your implementation
    print(confusion_matrix_(y, y_hat))
    # sklearn implementation
    print(confusion_matrix(y, y_hat))

    # Example 2:
    # your implementation
    print(confusion_matrix_(y, y_hat, labels=['dog', 'norminet']))
    # sklearn implementation
    print(confusion_matrix(y, y_hat, labels=['dog', 'norminet']))

    # Example 3:
    print(confusion_matrix_(y, y_hat, df_option=True))

    # Example 4:
    print(confusion_matrix_(y, y_hat, labels=['bird', 'dog'], df_option=True))
