import numpy as np
from numpy import ndarray


class MyLogisticRegression():
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """

    def _typechecker_construct(fun):
        def wrapper(self, theta, alpha=0.001, max_iter=1000):
            if (isinstance(theta, ndarray) and theta.size and
                    isinstance(alpha, float) and isinstance(max_iter, int)):
                theta = theta.reshape(
                    (-1, 1)) if theta.ndim == 1 else theta
                if theta.ndim == 2 and theta.shape[1] == 1:
                    return fun(self, theta, alpha, max_iter)
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
                        x.shape[1] == self.theta.shape[0] - 1 and y.shape == (x.shape[0], 1)):
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
                            x.shape[1] == self.theta.shape[0] - 1):
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

    def _sigmoid(self, x: ndarray) -> ndarray:
        return 1 / (1 + np.exp(-x))

    @_typechecker_construct
    def __init__(self, theta: ndarray, alpha: float = 0.001, max_iter: int = 1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = theta

    @_typechecker_fit
    def fit_(self, x: ndarray, y: ndarray) -> ndarray | None:
        x1 = self._add_intercept(x)
        m = y.shape[0]
        g2 = np.dot(x1.T, -y) / m
        for _ in range(self.max_iter):
            y_hat = self._sigmoid(np.dot(x1, self.theta))
            g1 = np.dot(x1.T, y_hat) / m
            new_theta = self.theta - self.alpha * (g1 + g2)
            if np.array_equal(new_theta, self.theta):
                break
            self.theta = new_theta

    @_typechecker_predict
    def predict_(self, x: ndarray) -> ndarray | None:
        x1 = self._add_intercept(x)
        return self._sigmoid(np.dot(x1, self.theta))

    @_typechecker_same
    def loss_elem_(self, y: ndarray, y_hat: ndarray) -> ndarray | None:
        eps = 1e-15
        return y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)

    @_typechecker_same
    def loss_(self, y: ndarray, y_hat: ndarray) -> float | None:
        y, y_hat, eps = y.reshape(-1), y_hat.reshape(-1), 1e-15
        return (np.dot(y, np.log(y_hat + eps)) + np.dot(1 - y, np.log(1 - y_hat + eps))) / -y.shape[0]


if __name__ == "__main__":
    MyLR = MyLogisticRegression
    X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [3., 5., 9., 14.]])
    Y = np.array([[1], [0], [1]])
    theta = np.array([[2], [0.5], [7.1], [-4.3], [2.09]])
    mylr = MyLR(theta)
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
