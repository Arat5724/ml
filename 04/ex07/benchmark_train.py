import numpy as np
from numpy import ndarray
import pandas as pd
from random import randint
from ridge import MyRidge as MyRidge
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error
import pickle
import time


def add_polynomial_features(x: ndarray, power: int) -> ndarray | None:
    x = x.reshape(-1)
    v = np.empty((power, x.shape[0]), dtype=x.dtype)
    v[0] = x
    for i in range(1, power):
        v[i] = v[i - 1] * x
    return v.T


def minmax(x: ndarray) -> ndarray:
    return (x - x.min())/(x.max() - x.min())


if __name__ == "__main__":
    data = pd.read_csv("space_avocado.csv")
    mylist = ["weight", "prod_distance", "time_delivery"]
    for feature in mylist:
        data[feature] = minmax(data[feature])
    X = np.array(data[mylist])
    Y = np.array(data["target"]).reshape(-1, 1)
    X, cvX, testX, Y, cvY, testY = (
        X[:2800], X[2800:4000], X[4000:], Y[:2800], Y[2800:4000], Y[4000:])

    mydict = {}
    thetas = [[Y.mean()], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
              [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]

    def add_polynomial_features3(data):
        return [add_polynomial_features(data[..., i], 4) for i in range(3)]

    X0, X1, X2 = add_polynomial_features3(X)
    cvX0, cvX1, cvX2 = add_polynomial_features3(cvX)
    testX0, testX1, testX2 = add_polynomial_features3(testX)

    def _():
        for k in range(1, 5):
            yield (2, 4, k)

    for a, b, c in _():
        key = a * 100 + b * 10 + c
        if key not in mydict:
            mydict[key] = {}
        X = np.hstack((X0[..., :a], X1[..., :b], X2[..., :c]))
        cvX = np.hstack(
            (cvX0[..., :a], cvX1[..., :b], cvX2[..., :c]))
        for i in range(6):
            lambda_ = i / 5
            print(key, lambda_)
            theta = np.array(thetas[:a + b + c + 1])
            mylr = MyRidge(theta, max_iter=1000000, lambda_=lambda_)
            mylr.fit_(X, Y)
            cvY_pred = mylr.predict_(cvX)
            mse = mylr.loss_(cvY, cvY_pred) * 2
            if i == 0 or mse < mydict[key]['mse']:
                mydict[key] = {
                    'lambda_': lambda_,
                    'mse': mse,
                    'theta': mylr.thetas
                }
        model = mydict[key]
        mylr = MyRidge(model['theta'], max_iter=1000000,
                       lambda_=model['lambda_'])
        testX = np.hstack(
            (testX0[..., :a], testX1[..., :b], testX2[..., :c]))
        testY_pred = mylr.predict_(testX)
        model['mse'] = mylr.loss_(testY, testY_pred) * 2
        model['theta'] = mylr.thetas

    with open("models.pickle", "wb") as f:
        pickle.dump(mydict, f)
