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


def add_intercept(x: ndarray) -> ndarray | None:
    return np.hstack((np.ones((x.shape[0], 1)), x))


def predict_(x: ndarray, theta: ndarray) -> ndarray | None:
    x1 = add_intercept(x)
    return np.dot(x1, theta)


def gradient(x: ndarray, y: ndarray, theta: ndarray) -> ndarray | None:
    x1 = add_intercept(x)
    y_hat = np.dot(x1, theta)
    return np.dot(x1.T, y_hat - y) / x.shape[0]


@typechecker
def fit_(x: ndarray, y: ndarray, theta: ndarray, alpha: float, max_iter: int) -> ndarray | None:
    """
    Description:
        Fits the model to the training dataset contained in x and y.
    Args:
         x: has to be a numpy.array, a matrix of dimension m * n:
                        (number of training examples, number of features).
         y: has to be a numpy.array, a vector of dimension m * 1:
                        (number of training examples, 1).
         theta: has to be a numpy.array, a vector of dimension (n + 1) * 1:
                        (number of features + 1, 1).
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the gradient descent
    Return:
        new_theta: numpy.array, a vector of dimension (number of features + 1, 1).
        None if there is a matching dimension problem.
        None if x, y, theta, alpha or max_iter is not of expected type.
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
    x = np.array([[0.2, 2., 20.], [0.4, 4., 40.],
                 [0.6, 6., 60.], [0.8, 8., 80.]])
    y = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
    theta = np.array([[42.], [1.], [1.], [1.]])
    # Example 0:
    theta2 = fit_(x, y, theta, alpha=0.0005, max_iter=42000)
    print(theta2)

    # Example 1:
    print(predict_(x, theta2))
