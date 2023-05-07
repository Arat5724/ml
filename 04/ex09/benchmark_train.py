import numpy as np
from numpy import ndarray
import pandas as pd
from my_logistic_regression import MyLogisticRegression as MyLR
from time import time


def add_polynomial_features(x: ndarray, power: int) -> ndarray:
    (m, n) = x.shape
    v = np.empty((n * power, m), dtype=x.dtype)
    xT = x.T
    v[:n] = xT
    for i in range(1, power):
        v[n * i: n * (i + 1)] = v[n * (i - 1): n * i] * xT
    return v.T


def minmax(x: ndarray) -> ndarray:
    return (x - x.min())/(x.max() - x.min())


def precision_score_(y: ndarray, y_hat: ndarray, pos_label: str | int = 1) -> float | None:
    y = y[(y_hat == pos_label)]
    return (y == pos_label).sum() / y.size


def recall_score_(y: ndarray, y_hat: ndarray, pos_label: str | int = 1) -> float | None:
    y_hat = y_hat[(y == pos_label)]
    return (y_hat == pos_label).sum() / y_hat.size


def f1_score_(y: ndarray, y_hat: ndarray, pos_label: str | int = 1) -> float | None:
    p = precision_score_(y, y_hat, pos_label)
    r = recall_score_(y, y_hat, pos_label)
    return 2 * p * r / (p + r)


if __name__ == "__main__":
    start = time()
    # 1
    mylist = ["weight", "height", "bone_density"]
    X = pd.read_csv("solar_system_census.csv")
    for feature in mylist:
        X[feature] = minmax(X[feature])
    X = np.array(X[mylist])

    X = add_polynomial_features(X, 3)
    Origin = np.array(pd.read_csv("solar_system_census_planets.csv")
                      ["Origin"]).reshape((-1, 1))
    trainingX, cvX, trainingOrigin, cvOrigin = (
        X[:80], X[80:100], Origin[:80], Origin[80:100])
    # zipcode = 0
    theta = np.array([[0.0], [0.0], [0.0], [0.0], [0.0],
                      [0.0], [0.0], [0.0], [0.0], [0.0]])
    mylr = MyLR(theta, max_iter=10000000, lambda_=0.0)
    trainingY = (trainingOrigin == 0) * 1
    cvY = (cvOrigin == 0) * 1
    mylr.fit_(trainingX, trainingY)
    cvY_pred = (mylr.predict_(cvX) >= 0.5) * 1
    print(cvY.reshape(-1))
    print(cvY_pred.reshape(-1))
    print(f1_score_(cvY, cvY_pred, 1))
    print(time() - start)
