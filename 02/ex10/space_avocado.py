import numpy as np
from numpy import ndarray
import pandas as pd
from random import randint
from mylinearregression import MyLinearRegression as MyLR
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error
import pickle


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
    with open("models.pickle", "rb") as f:
        mydict = pickle.load(f)
    print(mydict)
    minkey, minmse, mintheta = None, None, None
    for key, value in mydict.items():
        mse = value['mse']
        theta = value['theta']
        if minmse == None or mse < minmse:
            minkey, minmse, mintheta = key, mse, theta
    print(minkey, minmse, mintheta)
    i, j, k = minkey // 100, minkey % 100 // 10, minkey % 10

    print((i, j, k))
    data = pd.read_csv("space_avocado.csv")
    mylist = ["weight", "prod_distance", "time_delivery"]
    # 242
    for feature in mylist:
        data[feature] = minmax(data[feature])
    X = np.array(data[mylist])
    Y = np.array(data["target"]).reshape(-1, 1)
    x0 = add_polynomial_features(X[..., 0], 4)
    x1 = add_polynomial_features(X[..., 1], 4)
    x2 = add_polynomial_features(X[..., 2], 4)
    xijk = np.hstack((x0[..., :i], x1[..., :j], x2[..., :k]))

    mylr = MyLR(mintheta)
    print(mylr.gradient(xijk, Y).tolist())
    Y_model = mylr.predict_(xijk)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    # print(Y_model)
    for i, (feature, ax) in enumerate(zip(mylist, axes)):
        xi = X[..., i]
        ax.scatter(xi, Y_model, s=10, color=(0, 1, 0),
                   edgecolor='none', label="predicted")
        ax.scatter(xi, Y, s=5, color=(1, 0, 0),
                   edgecolor='none', label="real")
        ax.set_xlabel(feature)
        ax.set_ylabel("target")
        ax.legend()
    plt.show()
