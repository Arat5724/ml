import numpy as np
from numpy import ndarray


class MyLogisticRegression():
    """
    Description:
        My personnal logistic regression to classify things.
    """

    supported_penalties = ['l2']

    def __init__(self, theta: ndarray, alpha: float = 0.001, max_iter: int = 10000,
                 penalty='l2', lambda_: float = 1.0):
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = theta
        self.penalty = penalty
        self.lambda_ = lambda_ if penalty in self.supported_penalties else 0
        if self.theta.ndim == 1:
            self.theta = self.theta.reshape((*self.theta.shape, 1))

    def __sigmoid(x):
        return 1 / (1 + np.exp(-x))

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
        g2 = np.dot(x.T, -y).reshape((-1, 1)) / m
        for _ in range(self.max_iter):
            y_hat = MyLogisticRegression.__sigmoid(np.dot(x, self.theta))
            g1 = np.dot(x.T, y_hat).reshape((-1, 1)) / m
            new_theta = self.theta - self.alpha * (g1 + g2)
            if np.array_equal(new_theta, self.theta):
                break
            self.theta = new_theta

    def predict_(self, x: ndarray) -> ndarray:
        m, n = x.shape
        x1 = np.hstack((np.ones((m, 1)), x))
        return MyLogisticRegression.__sigmoid(np.dot(x1, self.theta))

    def loss_elem_(self, y: ndarray, y_hat: ndarray) -> ndarray:
        m, n = y.shape
        eps = 1e-15
        return y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)

    def loss_(self, y: ndarray, y_hat: ndarray) -> float:
        m, n = y.shape
        y, y_hat, eps = y.reshape(-1), y_hat.reshape(-1), 1e-15
        return (np.dot(y, np.log(y_hat + eps)) + np.dot(1 - y, np.log(1 - y_hat + eps))) / -m
