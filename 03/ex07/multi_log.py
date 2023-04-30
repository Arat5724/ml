import numpy as np
from numpy import ndarray
import pandas as pd
from my_logistic_regression import MyLogisticRegression as MyLR
import matplotlib.pyplot as plt
import sys


def probability(X, Origin, zipcode):
    Y = (Origin == zipcode)
    mylr = MyLR(np.array([[1.0], [1.0], [1.0], [1.0]]), max_iter=1000000)
    mylr.fit_(X, Y * 1)
    print(mylr.thetas)
    return mylr.predict_(X)


if __name__ == "__main__":
    X = np.array(pd.read_csv("solar_system_census.csv")
                 [["weight", "height", "bone_density"]])
    Origin = np.array(pd.read_csv("solar_system_census_planets.csv")
                      ["Origin"]).reshape(-1, 1)
    probabilities = np.hstack([probability(X, Origin, zipcode)
                               for zipcode in range(4)])
    prediction = probabilities.argmax(axis=1).reshape(-1, 1)

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    YY = X[(prediction == Origin).reshape(-1)]
    # YY = X
    YYc = Origin[(prediction == Origin).reshape(-1)]
    # YYc = prediction
    YYc = YYc.reshape(-1).tolist()
    for i, c in enumerate(YYc):
        tmp = [0, 0, 0]
        c = int(c)
        if c < 3:
            tmp[c] = 1
        YYc[i] = tmp
    NN = X[(prediction != Origin).reshape(-1)]

    m, n = X.shape

    ax.scatter(YY[..., 0], YY[..., 1], YY[..., 2], c=YYc)
    ax.scatter(NN[..., 0], NN[..., 1], NN[..., 2], color=(1, 1, 1))

    ax.set_xlabel("weight")
    ax.set_ylabel("height")
    ax.set_zlabel("bone_density")
    plt.show()
