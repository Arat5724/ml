import numpy as np
from numpy import ndarray


def typechecker(fun):
    def reshape_(*arg):
        return tuple(x.reshape((-1, 1)) if x.ndim == 1 else x for x in arg)

    def wrapper(x, theta):
        try:
            if isinstance(x, ndarray) and isinstance(theta, ndarray):
                x, theta = reshape_(x, theta)
                if (x.size and theta.size and
                    x.ndim == 2 and theta.ndim == 2 and
                        x.shape[1] == 1 and theta.shape == (2, 1)):
                    return fun(x, theta)
        except Exception as e:
            print(e)
    return wrapper


@typechecker
def simple_predict(x: ndarray, theta: ndarray) -> ndarray | None:
    """
    Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * 1.
        theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
        None if x or theta are empty numpy.ndarray.
        None if x or theta dimensions are not appropriate.
    Raises:
        This function should not raise any Exception.
    """
    return theta[0][0] + theta[1][0] * x


if __name__ == "__main__":
    x = np.arange(1, 6)
    # Example 1:
    theta1 = np.array([5, 0])
    print(simple_predict(x, theta1))

    # Example 2:
    theta2 = np.array([0, 1])
    print(simple_predict(x, theta2))

    # Example 3:
    theta3 = np.array([5, 3])
    print(simple_predict(x, theta3))

    # Example 4:
    theta4 = np.array([-3, 1])
    print(simple_predict(x, theta4))
