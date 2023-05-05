import numpy as np
from numpy import ndarray


class MyLinearRegression():
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """

    def _typechecker_construct(fun):
        def wrapper(self, thetas, alpha=0.001, max_iter=1000):
            if (isinstance(thetas, ndarray) and thetas.size and
                    isinstance(alpha, float) and isinstance(max_iter, int)):
                thetas = thetas.reshape(
                    (-1, 1)) if thetas.ndim == 1 else thetas
                if thetas.ndim == 2 and thetas.shape[1] == 1:
                    return fun(self, thetas, alpha, max_iter)
            else:
                raise TypeError("Invalid type")
        return wrapper

    def _typechecker_fit(fun):
        def reshape_(*arg):
            return tuple(x.reshape((-1, 1)) if x.ndim == 1 else x for x in arg)

        def wrapper(self, x, y):
            try:
                if (isinstance(x, ndarray) and isinstance(y, ndarray)):
                    x, y = reshape_(x, y)
                if (x.size and x.ndim == 2 and
                        x.shape[1] == self.thetas.shape[0] - 1 and y.shape == (x.shape[0], 1)):
                    return fun(self, x, y)
            except Exception as e:
                print(e)
        return wrapper

    def _typechecker_predict(fun):
        def wrapper(self, x):
            try:
                if isinstance(x, ndarray):
                    x, x.reshape((-1, 1)) if x.ndim == 1 else x
                    if (x.size and x.ndim == 2 and
                            x.shape[1] == self.thetas.shape[0] - 1):
                        return fun(self, x)
            except Exception as e:
                print(e)
        return wrapper

    def _typechecker_same(fun):
        def reshape_(*arg):
            return tuple(x.reshape((-1, 1)) if x.ndim == 1 else x for x in arg)

        def wrapper(self, y, y_hat):
            try:
                if isinstance(y, ndarray) and isinstance(y_hat, ndarray):
                    y, y_hat = reshape_(y, y_hat)
                    if (y.size and y.ndim == 2 and
                            y.shape[1] == 1 and y.shape == y_hat.shape):
                        return fun(self, y, y_hat)
            except Exception as e:
                print(e)
        return wrapper

    def _add_intercept(self, x: ndarray) -> ndarray:
        return np.hstack((np.ones((x.shape[0], 1)), x))

    @_typechecker_construct
    def __init__(self, thetas: ndarray, alpha: float = 0.001, max_iter: int = 1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas

    @_typechecker_fit
    def fit_(self, x: ndarray, y: ndarray) -> ndarray | None:
        x1 = self._add_intercept(x)
        m = y.shape[0]
        g2 = np.dot(x1.T, -y) / m
        for _ in range(self.max_iter):
            y_hat = np.dot(x1, self.thetas)
            g1 = np.dot(x1.T, y_hat) / m
            new_theta = self.thetas - self.alpha * (g1 + g2)
            if np.array_equal(new_theta, self.thetas):
                break
            self.thetas = new_theta

    @_typechecker_predict
    def predict_(self, x: ndarray) -> ndarray | None:
        x1 = self._add_intercept(x)
        return np.dot(x1, self.thetas)

    @_typechecker_same
    def loss_elem_(self, y: ndarray, y_hat: ndarray) -> ndarray | None:
        res = y_hat - y
        return res * res

    @_typechecker_same
    def loss_(self, y: ndarray, y_hat: ndarray) -> float | None:
        res = (y_hat - y).reshape(-1)
        return np.dot(res, res) / 2 / y.shape[0]
