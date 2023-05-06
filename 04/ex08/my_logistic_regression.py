import numpy as np
from numpy import ndarray
from inspect import signature


class MyLogisticRegression():
    """
    Description:
        My personnal logistic regression to classify things.
    """

    def _typechecker(fun):
        def wrapper(*args, **kwargs):
            bound_args = signature(fun).bind(*args, **kwargs).arguments
            for param, value in bound_args.items():
                if param in fun.__annotations__ and not isinstance(value, fun.__annotations__[param]):
                    return None
                if isinstance(value, ndarray) and (value.size == 0 or value.ndim != 2):
                    return None
            theta: ndarray = getattr(bound_args['self'], 'theta', None)
            x: ndarray = bound_args['x'] if 'x' in bound_args else None
            y: ndarray = bound_args['y'] if 'y' in bound_args else None
            y_hat: ndarray = bound_args['y_hat'] if 'y_hat' in bound_args else None
            if x is not None and x.shape[1] + 1 != theta.shape[0]:
                return None
            if y is not None and ((x is not None and y.shape[0] != x.shape[0]) or y.shape[1] != 1):
                return None
            if y_hat is not None:
                if y is not None:
                    if y_hat.shape != y.shape:
                        return None
                elif (x is not None and y_hat.shape[0] != x.shape[0]) or y_hat.shape[1] != 1:
                    return None
            return fun(*args, **kwargs)
        return wrapper

    def _add_intercept(self, x: ndarray) -> ndarray:
        return np.hstack((np.ones((x.shape[0], 1)), x))

    def _sigmoid(self, x: ndarray) -> ndarray:
        return 1 / (1 + np.exp(-x))

    @property
    def theta1(self):
        theta1 = self.theta.copy()
        theta1[0] = 0
        return theta1

    @property
    def l2(self) -> float:
        theta1 = self.theta1.reshape(-1)
        return np.dot(theta1, theta1)

    supported_penalties = ['l2']

    @_typechecker
    def __init__(self, theta: ndarray, alpha: float | int = 0.001, max_iter: int = 1000,
                 penalty: str = 'l2', lambda_: float | int = 1.0):
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = theta
        self.penalty = penalty
        self.lambda_ = lambda_ if penalty in self.supported_penalties else 0

    @_typechecker
    def fit_(self, x: ndarray, y: ndarray) -> ndarray | None:
        x1 = self._add_intercept(x)
        m = y.shape[0]
        g2 = np.dot(x1.T, -y) / m
        for _ in range(self.max_iter):
            y_hat = self._sigmoid(np.dot(x1, self.theta))
            g1 = (np.dot(x1.T, y_hat) + self.lambda_ * self.theta1) / m
            new_theta = self.theta - self.alpha * (g1 + g2)
            if np.array_equal(new_theta, self.theta):
                break
            self.theta = new_theta

    @_typechecker
    def predict_(self, x: ndarray) -> ndarray | None:
        x1 = self._add_intercept(x)
        return self._sigmoid(np.dot(x1, self.theta))

    @_typechecker
    def loss_elem_(self, y: ndarray, y_hat: ndarray) -> ndarray | None:
        eps = 1e-15
        return y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)

    @_typechecker
    def loss_(self, y: ndarray, y_hat: ndarray) -> float | None:
        y, y_hat, eps = y.reshape(-1), y_hat.reshape(-1), 1e-15
        return (-(y.dot(np.log(y_hat + eps)) + (1 - y).dot(np.log(1 - y_hat + eps))) +
                + self.lambda_ * self.l2 / 2) / y.shape[0]


if __name__ == "__main__":
    MyLR = MyLogisticRegression
    X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [3., 5., 9., 14.]])
    Y = np.array([[1], [0], [1]])
    theta = np.array([[2], [0.5], [7.1], [-4.3], [2.09]])
    mylr = MyLR(theta, lambda_=0)
    # Example 0:
    y_hat = mylr.predict_(X)
    print(y_hat)

    # Example 1:
    print(mylr.loss_(Y, y_hat))

    # Example 2:
    mylr.fit_(X, Y)
    print(mylr.theta)

    # Example 3:
    y_hat = mylr.predict_(X)
    print(y_hat)

    # Example 4:
    print(mylr.loss_(Y, y_hat))
