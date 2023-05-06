import numpy as np
from numpy import ndarray
from inspect import signature


class MyRidge():
    """
    Description:
        My personnal ridge regression class to fit like a boss.
    """

    def _typechecker(fun):
        def wrapper(*args, **kwargs):
            bound_args = signature(fun).bind(*args, **kwargs).arguments
            for param, value in bound_args.items():
                if param in fun.__annotations__ and not isinstance(value, fun.__annotations__[param]):
                    return None
                if isinstance(value, ndarray) and (value.size == 0 or value.ndim != 2):
                    return None
            thetas: ndarray = getattr(bound_args['self'], 'thetas', None)
            x: ndarray = bound_args['x'] if 'x' in bound_args else None
            y: ndarray = bound_args['y'] if 'y' in bound_args else None
            y_hat: ndarray = bound_args['y_hat'] if 'y_hat' in bound_args else None
            if x is not None and x.shape[1] + 1 != thetas.shape[0]:
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

    @_typechecker
    def __init__(self, thetas: ndarray, alpha: float | int = 0.001,
                 max_iter: int = 1000, lambda_: float | int = 0.5):
        self.thetas = thetas
        self.alpha = alpha
        self.max_iter = max_iter
        self.lambda_ = lambda_

    @property
    def theta1(self):
        theta1 = self.thetas.copy()
        theta1[0] = 0
        return theta1

    def get_params_(self):
        return self.__dict__

    def set_params_(self, params: dict):
        """
        Return
            True: succeeded
            False: failed
        """
        if not isinstance(params, dict):
            return False
        attrtypes = {
            "thetas": ndarray,
            "alpha": float,
            "max_iter": int,
            "lambda_": float
        }
        for k, v in params.items():
            if not (k in attrtypes and isinstance(v, attrtypes[k])):
                return False
        self.__dict__.update(params)
        return True

    @_typechecker
    def fit_(self, x: ndarray, y: ndarray) -> ndarray | None:
        x1 = self._add_intercept(x)
        m = y.shape[0]
        g2 = np.dot(x1.T, -y) / m
        for _ in range(self.max_iter):
            y_hat = np.dot(x1, self.thetas)
            g1 = (np.dot(x1.T, y_hat) + self.lambda_ * self.theta1) / m
            new_theta = self.thetas - self.alpha * (g1 + g2)
            if np.array_equal(new_theta, self.thetas):
                break
            self.thetas = new_theta

    @_typechecker
    def predict_(self, x: ndarray) -> ndarray | None:
        x1 = self._add_intercept(x)
        return np.dot(x1, self.thetas)

    @_typechecker
    def gradient_(self, y: ndarray, x: ndarray):
        m, n = x.shape
        x1 = self._add_intercept(x)
        y_hat = np.dot(x1, self.thetas)
        theta1 = self.theta1
        return (np.dot(x1.T, y_hat - y) + self.lambda_ * theta1) / m

    @_typechecker
    def loss_elem_(self, y: ndarray, y_hat: ndarray) -> ndarray | None:
        return (y_hat - y) ** 2

    @_typechecker
    def loss_(self, y: ndarray, y_hat: ndarray) -> float | None:
        ysub, theta1 = (y_hat - y).reshape(-1), self.theta1.reshape(-1)
        return (np.dot(ysub, ysub) + self.lambda_ * np.dot(theta1, theta1)) / 2 / y.shape[0]
