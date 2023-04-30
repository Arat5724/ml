import numpy as np
from numpy import ndarray
import pandas as pd
from random import randint
from ridge import MyRidge as MyRidge
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error
import pickle
import time


def data_spliter(x, y, proportion):
    m, _ = x.shape
    x, y = np.copy(x), np.copy(y)
    for i in range(m):
        j = randint(0, i)
        x[[i, j]] = x[[j, i]]
        y[[i, j]] = y[[j, i]]
    m0 = int(m * proportion)
    return x[:m0], x[m0:], y[:m0], y[m0:]


def add_polynomial_features(x: ndarray, power: int):
    # power must greater than or equal to 0
    x = x.reshape(-1)
    (m,) = x.shape
    v = np.empty((power, m), dtype=x.dtype)
    v[0] = x
    for i in range(1, power):
        v[i] = v[i - 1] * x
    return np.transpose(v)


def minmax(x: ndarray) -> ndarray:
    return (x - x.min())/(x.max() - x.min())


if __name__ == "__main__":
    data = pd.read_csv("space_avocado.csv")
    mylist = ["weight", "prod_distance", "time_delivery"]
    for feature in mylist:
        data[feature] = minmax(data[feature])
    X = np.array(data[mylist])
    Y = np.array(data["target"]).reshape(-1, 1)
    X, testX, Y, testY = X[:2450], X[2450:], Y[:2450], Y[2450:]
    with open("models.pickle", "rb") as f:
        mydict = pickle.load(f)
    # mydict = {}
    x0 = add_polynomial_features(X[..., 0], 4)
    x1 = add_polynomial_features(X[..., 1], 4)
    x2 = add_polynomial_features(X[..., 2], 4)
    testx0 = add_polynomial_features(testX[..., 0], 4)
    testx1 = add_polynomial_features(testX[..., 1], 4)
    testx2 = add_polynomial_features(testX[..., 2], 4)
    thetas = [[Y.mean()], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
              [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
    start = time.time()
    for i in range(1, 5):
        for j in range(1, 5):
            for k in range(1, 5):
                print((i, j, k))
                xijk = np.hstack((x0[..., :i], x1[..., :j], x2[..., :k]))
                testxijk = np.hstack(
                    (testx0[..., :i], testx1[..., :j], testx2[..., :k]))
                # myridge = MyRidge(np.array(thetas[:i+j+k+1]), max_iter=100000)
                myridge = MyRidge(mydict[i * 100 + j * 10 + k]
                                  ['theta'], max_iter=100000)
                myridge.fit_(xijk, Y)
                Y_model = myridge.predict_(testxijk)
                mydict[i * 100 + j * 10 + k] = {
                    'mse': myridge.loss_(testY, Y_model),
                    'theta': myridge.thetas
                }

    with open("models.pickle", "wb") as f:
        pickle.dump(mydict, f)
