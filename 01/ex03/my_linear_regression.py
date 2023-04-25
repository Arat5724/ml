import numpy as np
from numpy import ndarray


class MyLinearRegression():
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """

    def __init__(self, thetas: ndarray, alpha: float = 0.001, max_iter: int = 1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas
        if self.thetas.ndim == 1:
            self.thetas = self.thetas.reshape((*self.thetas.shape, 1))

    def __add_intercept(self, x: ndarray) -> ndarray:
        if x.ndim == 1:
            x = x.reshape((*x.shape, 1))
        m, n = x.shape
        if m * n == 0:
            return None
        return np.hstack((np.ones((m, 1)), x))

    def fit_(self, x: ndarray, y: ndarray) -> ndarray:
        x = self.__add_intercept(x)
        if y.ndim == 1:
            y = y.reshape((*y.shape, 1))
        m, _ = y.shape
        g2 = np.dot(x.T, -y).reshape((2, 1)) / m
        for _ in range(self.max_iter):
            y_hat = np.dot(x, self.thetas)
            g1 = np.dot(x.T, y_hat).reshape((2, 1)) / m
            new_theta = self.thetas - self.alpha * (g1 + g2)
            if np.array_equal(new_theta, self.thetas):
                break
            self.thetas = new_theta

    def predict_(self, x: ndarray) -> ndarray:
        x = self.__add_intercept(x)
        if x is None:
            return None
        return np.dot(x, self.thetas)

    def loss_elem_(self, y: ndarray, y_hat: ndarray) -> ndarray:
        res = y_hat - y
        return res * res

    def loss_(self, y: ndarray, y_hat: ndarray) -> float:
        (m, _) = y.shape
        res = (y_hat - y).reshape(-1)
        return np.dot(res, res) / 2 / m
