import numpy as np
from numpy import ndarray


class MyLinearRegression():
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """

    def _typechecker(fun):
        def wrapper(*args, **kwargs):
            from inspect import signature
            bound_args = signature(fun).bind(*args, **kwargs).arguments
            for param, value in bound_args.items():
                if param in fun.__annotations__ and not isinstance(value, fun.__annotations__[param]):
                    return None
                if isinstance(value, ndarray) and (value.size == 0 or value.ndim != 2):
                    return None
            thetas = (bound_args['thetas'] if 'thetas' in bound_args else
                      getattr(bound_args['self'], 'thetas', None))
            x, y, y_hat = [bound_args[ele] if ele in bound_args else None
                           for ele in ['x', 'y', 'y_hat']]
            if ((thetas is not None and ((x is not None and thetas.shape[0] != x.shape[1] + 1) or thetas.shape[1] != 1)) or
                (y is not None and ((x is not None and y.shape[0] != x.shape[0]) or y.shape[1] != 1)) or
                (y_hat is not None and ((y is not None and y_hat.shape != y.shape) or
                                        (y is None and ((x is not None and y_hat.shape[0] != x.shape[0]) or y_hat.shape[1] != 1))))):
                return None
            try:
                return fun(*args, **kwargs)
            except:
                return None
        return wrapper

    def _add_intercept(self, x: ndarray) -> ndarray:
        return np.hstack((np.ones((x.shape[0], 1)), x))

    @_typechecker
    def __init__(self, thetas: ndarray, alpha: float | int = 0.001, max_iter: int = 1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas

    @_typechecker
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

    @_typechecker
    def predict_(self, x: ndarray) -> ndarray | None:
        x1 = self._add_intercept(x)
        return np.dot(x1, self.thetas)

    @_typechecker
    def loss_elem_(self, y: ndarray, y_hat: ndarray) -> ndarray | None:
        res = y_hat - y
        return res * res

    @_typechecker
    def loss_(self, y: ndarray, y_hat: ndarray) -> float | None:
        res = (y_hat - y).reshape(-1)
        return np.dot(res, res) / 2 / y.shape[0]
