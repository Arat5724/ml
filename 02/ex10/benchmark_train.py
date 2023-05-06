import numpy as np
from numpy import ndarray
import pandas as pd
from mylinearregression import MyLinearRegression as MyLR
from random import randint
import pickle


def data_spliter(x: ndarray, y: ndarray, proportion: float):
    m = x.shape[0]
    x, y = np.copy(x), np.copy(y)
    for i in range(m):
        j = randint(0, i)
        x[[i, j]] = x[[j, i]]
        y[[i, j]] = y[[j, i]]
    index = int(m * proportion)
    return x[:index], x[index:], y[:index], y[index:]


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
    X, testX, Y, testY = X[:2450], X[2450:], Y[:2450], Y[2450:]
    try:
        with open("models.pickle", "rb") as f:
            mydict = pickle.load(f)
    except:
        mydict = {}
        thetas = [[Y.mean()], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                  [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]

    X0 = add_polynomial_features(X[..., 0], 4)
    X1 = add_polynomial_features(X[..., 1], 4)
    X2 = add_polynomial_features(X[..., 2], 4)
    testX0 = add_polynomial_features(testX[..., 0], 4)
    testX1 = add_polynomial_features(testX[..., 1], 4)
    testX2 = add_polynomial_features(testX[..., 2], 4)

    def _():
        for i in range(1, 5):
            for j in range(1, 5):
                for k in range(1, 5):
                    yield (i, j, k)

    for i, j, k in _():
        key = i * 100 + j * 10 + k
        print(key)
        X = np.hstack((X0[..., :i], X1[..., :j], X2[..., :k]))
        testX = np.hstack((testX0[..., :i], testX1[..., :j], testX2[..., :k]))
        try:
            theta = mydict[key]['theta']
        except:
            theta = np.array(thetas[:i + j + k + 1])
        mylr = MyLR(theta, max_iter=1000000)
        mylr.fit_(X, Y)
        Y_model = mylr.predict_(testX)
        mydict[key] = {
            'mse': mylr.loss_(testY, Y_model) * 2,
            'theta': mylr.thetas
        }

    with open("models.pickle", "wb") as f:
        pickle.dump(mydict, f)
