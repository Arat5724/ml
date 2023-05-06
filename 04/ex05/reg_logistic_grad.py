import numpy as np
from numpy import ndarray


def typechecker(fun):
    def reshape_(*arg):
        return tuple(x.reshape((-1, 1)) if x.ndim == 1 else x for x in arg)

    def wrapper(y, x, theta, lambda_):
        try:
            if (isinstance(x, ndarray) and isinstance(y, ndarray) and
                    isinstance(theta, ndarray) and isinstance(lambda_, (float | int))):
                y, x, theta = reshape_(y, x, theta)
                if (x.size and x.ndim == 2 and
                        y.shape == (x.shape[0], 1) and theta.shape == (x.shape[1] + 1, 1)):
                    return fun(y, x, theta, lambda_)
        except Exception as e:
            print(e)
    return wrapper


def add_intercept(x: ndarray) -> ndarray:
    return np.hstack((np.ones((x.shape[0], 1)), x))


def sigmoid_(x: ndarray) -> ndarray:
    return 1 / (1 + np.exp(-x))


@typechecker
def reg_logistic_grad(y: ndarray, x: ndarray, theta: ndarray, lambda_: float) -> ndarray | None:
    """Computes the regularized logistic gradient of three non-empty numpy.ndarray,
    with two for-loops. The three arrays must have compatible shapes.
    Args:
        y: has to be a numpy.ndarray, a vector of shape m * 1.
        x: has to be a numpy.ndarray, a matrix of dimesion m * n.
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
        lambda_: has to be a float.
    Returns:
        A numpy.ndarray, a vector of shape n * 1, containing the results of the formula for all j.
        None if y, x, or theta are empty numpy.ndarray.
        None if y, x or theta does not share compatibles shapes.
    Raises:
        This function should not raise any Exception.
    """
    m, n = x.shape
    x1 = add_intercept(x)
    theta1 = np.copy(theta)
    theta1[0] = 0.0
    y_hat = sigmoid_(np.dot(x1, theta))
    ysub = (y_hat - y).reshape(-1)
    grad = np.zeros_like(theta)
    for j in range(n + 1):
        for i in range(m):
            grad[j] += ysub[i] * x1[i][j]
        grad[j] += lambda_ * theta1[j]
        grad[j] /= m
    return grad


def vec_reg_logistic_grad(y: ndarray, x: ndarray, theta: ndarray, lambda_: float) -> ndarray | None:
    """Computes the regularized logistic gradient of three non-empty numpy.ndarray,
        without any for-loop. The three arrays must have compatible shapes.
    Args:
        y: has to be a numpy.ndarray, a vector of shape m * 1.
        x: has to be a numpy.ndarray, a matrix of shape m * n.
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
        lambda_: has to be a float.
    Returns:
        A numpy.ndarray, a vector of shape n * 1, containing the results of the formula for all j.
        None if y, x, or theta are empty numpy.ndarray.
        None if y, x or theta does not share compatibles shapes.
    Raises:
        This function should not raise any Exception.
    """
    m, n = x.shape
    x1 = add_intercept(x)
    theta1 = np.copy(theta)
    theta1[0] = 0
    y_hat = sigmoid_(np.dot(x1, theta))
    return (np.dot(x1.T, y_hat - y) + lambda_ * theta1) / m


if __name__ == "__main__":
    x = np.array([[0, 2, 3, 4],
                  [2, 4, 5, 5],
                  [1, 3, 2, 7]])
    y = np.array([[0], [1], [1]])
    theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    # Example 1.1:
    print(reg_logistic_grad(y, x, theta, 1))

    # Example 1.2:
    print(vec_reg_logistic_grad(y, x, theta, 1))

    # Example 2.1:
    print(reg_logistic_grad(y, x, theta, 0.5))

    # Example 2.2:
    print(vec_reg_logistic_grad(y, x, theta, 0.5))

    # Example 3.1:
    print(reg_logistic_grad(y, x, theta, 0.0))

    # Example 3.2:
    print(vec_reg_logistic_grad(y, x, theta, 0.0))
