import numpy as np
from numpy import ndarray
import pandas as pd
from my_logistic_regression import MyLogisticRegression as MyLR
from time import time
import pickle


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
    if y.size == 0:
        return 0
    return (y == pos_label).sum() / y.size


def recall_score_(y: ndarray, y_hat: ndarray, pos_label: str | int = 1) -> float | None:
    y_hat = y_hat[(y == pos_label)]
    if y_hat.size == 0:
        return 0
    return (y_hat == pos_label).sum() / y_hat.size


def f1_score_(y: ndarray, y_hat: ndarray, pos_label: str | int = 1) -> float | None:
    p = precision_score_(y, y_hat, pos_label)
    r = recall_score_(y, y_hat, pos_label)
    if p + r == 0:
        return 0
    return 2 * p * r / (p + r)


def multiclass_f1_score_(y: ndarray, y_hat: ndarray) -> float | None:
    labels = np.unique(y)
    ret = 0.0
    for label in labels:
        y1 = (y == label) * 1
        y_hat1 = (y_hat == label) * 1
        count = y1.sum()
        ret += f1_score_(y1, y_hat1) * count
    return ret / y.size


if __name__ == "__main__":
    # 1
    try:
        with open("models.pickle", "rb") as f:
            mydict = pickle.load(f)
    except:
        mydict = {}
    mylist = ["weight", "height", "bone_density"]
    X = pd.read_csv("solar_system_census.csv")
    for feature in mylist:
        X[feature] = minmax(X[feature])
    X = np.array(X[mylist])
    X = add_polynomial_features(X, 3)
    Origin = np.array(pd.read_csv("solar_system_census_planets.csv")
                      ["Origin"]).reshape((-1, 1))
    trainingX, crossValidationX, trainingOrigin, crossValidationOrigin = (
        X[:60], X[60:100], Origin[:60], Origin[60:100])

    for i in range(6):
        lambda_ = i / 5
        print(f"lambda_: {lambda_}")
        if lambda_ not in mydict:
            mydict[lambda_] = {
                'models': [None] * 4
            }
        for zipcode in range(4):
            print(f"zipcode: {zipcode}")
            trainingY = (trainingOrigin == zipcode) * 1
            crossValidationY = (crossValidationOrigin == zipcode) * 1
            if (mylr := mydict[lambda_]['models'][zipcode]) is None:
                mylr = MyLR(
                    np.array([[0.0], [0.0], [0.0], [0.0], [0.0],
                              [0.0], [0.0], [0.0], [0.0], [0.0]]), max_iter=3000000, lambda_=lambda_)
            print(mylr.gradient_(trainingX, trainingY))
            mylr.fit_(trainingX, trainingY)
            print(mylr.gradient_(trainingX, trainingY))
            mydict[lambda_]['models'][zipcode] = mylr
        crossValidationY_pred = np.hstack(
            [model.predict_(crossValidationX) for model in mydict[lambda_]['models']]).argmax(axis=1).reshape((-1, 1))
        mydict[lambda_]['f1'] = multiclass_f1_score_(
            crossValidationY, crossValidationY_pred)
        print(crossValidationOrigin.reshape(-1))
        print(crossValidationY_pred.reshape(-1))
        print(mydict[lambda_]['f1'])

    with open("models.pickle", "wb") as f:
        pickle.dump(mydict, f)
