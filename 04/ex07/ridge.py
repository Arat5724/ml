import numpy as np
from numpy import ndarray


class MyRidge:
    """
    Description:
        My personnal ridge regression class to fit like a boss.
    """
    attrtypes = {
        "thetas": ndarray,
        "alpha": float,
        "max_iter": int,
        "lambda_": float
    }

    def __init__(self, thetas: ndarray, alpha: float = 0.001,
                 max_iter: int = 1000, lambda_: float = 0.5):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas
        self.lambda_ = lambda_
        # self.__thetas1 = self.thetas.copy()

        if self.thetas.ndim == 1:
            self.thetas = self.thetas.reshape((*self.thetas.shape, 1))

    def __add_intercept(self, x: ndarray) -> ndarray:
        if x.ndim == 1:
            x = x.reshape((*x.shape, 1))
        m, n = x.shape
        if m * n == 0:
            return None
        return np.hstack((np.ones((m, 1)), x))

    def get_params_(self):
        return self.__dict__

    def set_params_(self, params: dict):
        """
        Return
            True: succeeded
            False: failed
        """
        # TO_DO: check type
        for k, v in params.items():
            if not (k in MyRidge.attrtypes and isinstance(v, MyRidge.attrtypes[k])):
                print("invalid parameter")  # TO_DO
                return False
        self.__dict__.update(params)
        return True

    def loss_(self, y: ndarray, y_hat: ndarray) -> float:
        (m, _) = y.shape
        res, theta1 = (y_hat - y).reshape(-1), self.thetas.copy().reshape(-1)
        theta1[0] = 0.0
        return (np.dot(res, res) + self.lambda_ * np.dot(theta1, theta1)) / 2 / m

    def loss_elem_(self, y: ndarray, y_hat: ndarray) -> ndarray:
        res = y_hat - y
        theta1 = self.thetas.copy()
        theta1[0] = 0.0
        return res * res + self.lambda_ * theta1 * theta1

    def predict_(self, x: ndarray) -> ndarray:
        x = self.__add_intercept(x)
        if x is None:
            return None
        return np.dot(x, self.thetas)

    def gradient_(self, y, x):
        m, n = x.shape
        x1 = np.hstack((np.ones((m, 1), dtype=x.dtype), x))
        y_hat = np.dot(x1, self.thetas)
        theta1 = self.thetas.copy()
        theta1[0] = 0.0
        return (np.dot(x1.T, y_hat - y) + self.lambda_ * theta1) / m

    def fit_(self, x: ndarray, y: ndarray) -> ndarray:
        x = self.__add_intercept(x)
        if y.ndim == 1:
            y = y.reshape((*y.shape, 1))
        m, _ = y.shape
        theta1 = self.thetas.copy()
        theta1[0] = 0.0
        g2 = np.dot(x.T, -y).reshape((-1, 1)) / m
        for _ in range(self.max_iter):
            y_hat = np.dot(x, self.thetas)
            g1 = np.dot(x.T, y_hat).reshape((-1, 1)) / m
            g3 = self.lambda_ * theta1 / m
            new_theta = self.thetas - self.alpha * (g1 + g2 + g3)
            if np.array_equal(new_theta, self.thetas):
                break
            self.thetas = new_theta
