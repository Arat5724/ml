import numpy as np
from numpy import ndarray


def typechecker(fun):
    def reshape_(*arg):
        return tuple(x.reshape((-1, 1)) if x.ndim == 1 else x for x in arg)

    def wrapper(x, y, theta, alpha, max_iter):
        try:
            if (isinstance(x, ndarray) and isinstance(y, ndarray) and isinstance(theta, ndarray) and
                    isinstance(alpha, float) and isinstance(max_iter, int)):
                x, y, theta = reshape_(x, y, theta)
                if (x.size and theta.size and x.ndim == 2 and theta.ndim == 2 and
                        x.shape[1] == 1 and theta.shape == (2, 1) and x.shape == y.shape):
                    return fun(x, y, theta, alpha, max_iter)
        except Exception as e:
            print(e)
    return wrapper


def add_intercept(x: ndarray) -> ndarray | None:
    return np.hstack((np.ones((x.shape[0], 1)), x))


def predict_(x: ndarray, theta: ndarray) -> ndarray | None:
    x1 = add_intercept(x)
    return np.dot(x1, theta)


def gradient(x: ndarray, y: ndarray, theta: ndarray) -> ndarray | None:
    x1 = add_intercept(x)
    y_hat = np.dot(x1, theta)
    return np.dot(x1.T, y_hat - y) / y.shape[0]


@typechecker
def fit_(x: ndarray, y: ndarray, theta: ndarray, alpha: float, max_iter: int) -> ndarray | None:
    """
    Description:
        Fits the model to the training dataset contained in x and y.
    Args:
        x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the gradient descent
    Returns:
        new_theta: numpy.ndarray, a vector of dimension 2 * 1.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exception.
    """
    for _ in range(max_iter):
        new_theta = theta - alpha * gradient(x, y, theta)
        if np.array_equal(new_theta, theta):
            break
        theta = new_theta
    return theta


if __name__ == "__main__":
    x = np.array([[12.4956442], [21.5007972], [
                 31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [
                 45.7655287], [46.6793434], [59.5585554]])
    theta = np.array([1, 1]).reshape((-1, 1))

    theta1 = fit_(x, y, theta, alpha=5e-8, max_iter=1500000)
    print(theta1)
    print(predict_(x, theta1))
