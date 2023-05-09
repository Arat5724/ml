import numpy as np
from numpy import ndarray
import pandas as pd
from my_logistic_regression import MyLogisticRegression as MyLR
import matplotlib.pyplot as plt
import sys
from random import randint


def zc():
    usage = """usage
    python3 mono_log.py -zipcode=x
example
    python3 mono_log.py -zipcode=2
    """
    argv = sys.argv
    if len(argv) == 2 and argv[1].startswith("-zipcode=") and \
            (tmp := argv[1][9:]).isdigit() and (0 <= (zipcode := int(tmp)) <= 3):
        return zipcode
    print(usage)
    exit()


def data_spliter(x: ndarray, y: ndarray, proportion: float):
    m = x.shape[0]
    x, y = np.copy(x), np.copy(y)
    for i in range(m):
        j = randint(0, i)
        x[[i, j]] = x[[j, i]]
        y[[i, j]] = y[[j, i]]
    index = int(m * proportion)
    return x[:index], x[index:], y[:index], y[index:]


def plot2d(X, Y, prediction, mylist):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    correct = (Y == prediction).reshape(-1)
    wrong = ~correct
    TX, TY = X[correct], Y[correct]
    FX, FNY, FPY = X[wrong], Y[wrong], prediction[wrong]
    for i, (feature, ax) in enumerate(zip(mylist, axes)):
        ax.scatter(TX[..., i], TY, s=2,
                   color="black", label="correct prediction")
        ax.scatter(FX[..., i], FPY, s=2,
                   color="red", label="wrong prediction")
        ax.scatter(FX[..., i], FNY, s=2,
                   color="blue", label="true")
        ax.set_xlabel(feature)
        ax.set_ylabel("Area")
        ax.legend()


def plot3d(X, Y, prediction):
    ax = plt.figure().add_subplot(projection='3d')
    Y, prediction = Y == 1, prediction == 1
    # Ture positive
    TP = X[(Y & prediction).reshape(-1)]
    ax.scatter(TP[..., 0], TP[..., 1], TP[..., 2],
               color='white', label="Ture positive")
    # Ture negative
    TN = X[(~Y & ~prediction).reshape(-1)]
    ax.scatter(TN[..., 0], TN[..., 1], TN[..., 2],
               color='black', label="Ture negative")
    # False positive
    FP = X[(~Y & prediction).reshape(-1)]
    ax.scatter(FP[..., 0], FP[..., 1], FP[..., 2],
               color='red', label="False positive")
    # False negative
    FN = X[(Y & ~prediction).reshape(-1)]
    ax.scatter(FN[..., 0], FN[..., 1], FN[..., 2],
               color='blue', label="False negative")
    ax.legend()
    ax.set_xlabel("weight")
    ax.set_ylabel("height")
    ax.set_zlabel("bone_density")


def minmax(x: ndarray) -> ndarray:
    mins, maxs = x.min(axis=0), x.max(axis=0)
    return (x - mins) / (maxs - mins)


if __name__ == "__main__":
    zipcode = zc()
    mylist = ["weight", "height", "bone_density"]
    X = np.array(pd.read_csv("solar_system_census.csv")
                 [mylist])
    Origin = np.array(pd.read_csv("solar_system_census_planets.csv")
                      ["Origin"]).reshape(-1, 1)
    Y = (Origin == zipcode) * 1

    trainingX, testX, trainingY, testY = \
        data_spliter(X, Y, 0.5)

    mylr = MyLR(np.array([[0.0], [0.0], [0.0], [0.0]]), max_iter=1000000)
    mylr.fit_(trainingX, trainingY)
    testPrediction = (mylr.predict_(testX) >= 0.5) * 1
    prediction = (mylr.predict_(X) >= 0.5) * 1
    correct = (testPrediction == testY).sum()
    print("the fraction of correct predictions :",
          f"{correct} / {testPrediction.size}")

    plot2d(X, Y, prediction, mylist)
    plot3d(X, Y, prediction)

    plt.show()
