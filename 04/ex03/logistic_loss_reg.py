import numpy as np
from numpy import ndarray


def typechecker(fun):
    def reshape_(*arg):
        return tuple(x.reshape((-1, 1)) if x.ndim == 1 else x for x in arg)

    def wrapper(y, y_hat, theta, lambda_):
        try:
            if (isinstance(y, ndarray) and isinstance(y_hat, ndarray) and
                    isinstance(theta, ndarray) and isinstance(lambda_, (float | int))):
                y, y_hat, theta = reshape_(y, y_hat, theta)
                if (y.size and theta.size and y.ndim == theta.ndim == 2 and
                        y.shape[1] == theta.shape[1] == 1 and y.shape == y_hat.shape):
                    return fun(y, y_hat, theta, lambda_)
        except Exception as e:
            print(e)
    return wrapper


def l2(theta: ndarray) -> float:
    theta1 = theta[1:].reshape(-1)
    return np.dot(theta1, theta1)


@typechecker
def reg_log_loss_(y: ndarray, y_hat: ndarray, theta: ndarray, lambda_: float) -> float | None:
    """Computes the regularized loss of a logistic regression model from two non-empty numpy.ndarray, without any for loop
    Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
        lambda_: has to be a float.
    Returns:
        The regularized loss as a float.
        None if y, y_hat, or theta is empty numpy.ndarray.
        None if y and y_hat do not share the same shapes.
    Raises:
        This function should not raise any Exception.
    """
    m, n = y.shape
    y, y_hat = y.reshape(-1), y_hat.reshape(-1)
    return (-(y.dot(np.log(y_hat + 1e-15)) + (1 - y).dot(np.log(1 - y_hat + 1e-15))) +
            + lambda_ * l2(theta) / 2) / m


if __name__ == "__main__":
    y = np.array([1, 1, 0, 0, 1, 1, 0]).reshape((-1, 1))
    y_hat = np.array([.9, .79, .12, .04, .89, .93, .01]).reshape((-1, 1))
    theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))

    # Example :
    print(reg_log_loss_(y, y_hat, theta, .5))

    # Example :
    print(reg_log_loss_(y, y_hat, theta, .05))

    # Example :
    print(reg_log_loss_(y, y_hat, theta, .9))
