import numpy as np
from numpy import ndarray
import pandas as pd
from my_logistic_regression import MyLogisticRegression as MyLR
import matplotlib.pyplot as plt
import sys


def plot3d(Y, Y_model):
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    YY = X[(Y & Y_model).reshape(-1)]
    YN = X[(Y & ~Y_model).reshape(-1)]
    NY = X[(~Y & Y_model).reshape(-1)]
    NN = X[(~Y & ~Y_model).reshape(-1)]
    m, n = X.shape
    ax.scatter(YY[..., 0], YY[..., 1], YY[..., 2], color=(1, 1, 1))
    ax.scatter(NN[..., 0], NN[..., 1], NN[..., 2], color=(0, 0, 0))
    ax.scatter(NY[..., 0], NY[..., 1], NY[..., 2], color=(1, 0, 0))
    ax.scatter(YN[..., 0], YN[..., 1], YN[..., 2], color=(0, 0, 1))
    ax.set_xlabel("weight")
    ax.set_ylabel("height")
    ax.set_zlabel("bone_density")


if __name__ == "__main__":
    usage = """usage
    python3 mono_log.py -zipcode=x
example
    python3 mono_log.py -zipcode=2
    """

    argv = sys.argv
    if len(argv) != 2:
        print(usage)
        exit()
    if argv[1].startswith("-zipcode"):
        argv[1] = argv[1][8:]
    if not argv[1].isdigit():
        print(usage)
        exit()
    zipcode = int(argv[1])
    if not (0 <= zipcode <= 3):
        print(usage)
        exit()
    mylist = ["weight", "height", "bone_density"]
    X = np.array(pd.read_csv("solar_system_census.csv")
                 [mylist])
    Origin = np.array(pd.read_csv("solar_system_census_planets.csv")
                      ["Origin"]).reshape(-1, 1)
    Y = (Origin == zipcode)
    mylr = MyLR(np.array([[1.0], [1.0], [1.0], [1.0]]), max_iter=10000)
    mylr.fit_(X, Y * 1)
    Y_model = mylr.predict_(X)
    Y_model = Y_model >= 0.5

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, (feature, ax) in enumerate(zip(mylist, axes)):
        ax.scatter(X[..., i], Y, color=(1, 0, 0, 0.5))
        ax.scatter(X[..., i], Y_model * 1, color=(0, 1, 0, 0.5))

    plot3d(Y, Y_model)

    plt.show()
