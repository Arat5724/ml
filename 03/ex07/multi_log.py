import numpy as np
from numpy import ndarray
import pandas as pd
from my_logistic_regression import MyLogisticRegression as MyLR
import matplotlib.pyplot as plt
from random import randint


def data_spliter(x: ndarray, y: ndarray, proportion: float):
    m = x.shape[0]
    x, y = np.copy(x), np.copy(y)
    for i in range(m):
        j = randint(0, i)
        x[[i, j]] = x[[j, i]]
        y[[i, j]] = y[[j, i]]
    index = int(m * proportion)
    return x[:index], x[index:], y[:index], y[index:]


def probability(trainingX, trainingOrigin, X, testX, zipcode):
    mylr = MyLR(np.array([[0.0], [0.0], [0.0], [0.0]]), max_iter=1000000)
    trainingY = (trainingOrigin == zipcode)
    mylr.fit_(trainingX, trainingY * 1)
    return mylr.predict_(X), mylr.predict_(testX)


def plot2d(X, Origin, prediction, mylist):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    correct = (Origin == prediction).reshape(-1)
    wrong = ~correct
    TX, TY = X[correct], Origin[correct]
    FX, FNY, FPY = X[wrong], Origin[wrong], prediction[wrong]
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


def plot3d(X, Origin, prediction):
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

    areas = ["Venus", "Earth", "Mars", "Belt"]
    citizens = [X[(prediction == zipcode) & (Origin == zipcode)]
                for zipcode in range(4)]
    colors = ["yellow", "blue", "red", "black"]

    for area, citizen, color in zip(areas, citizens, colors):
        ax.scatter(citizen[..., 0], citizen[..., 1],
                   citizen[..., 2], c=color, label=area)

    wrong = X[prediction != Origin]
    ax.scatter(wrong[..., 0], wrong[..., 1], wrong[..., 2],
               color="green", label="wrong prediction")

    ax.legend()
    ax.set_xlabel("weight")
    ax.set_ylabel("height")
    ax.set_zlabel("bone_density")


if __name__ == "__main__":
    # 1
    mylist = ["weight", "height", "bone_density"]
    X = np.array(pd.read_csv("solar_system_census.csv")[mylist])
    Origin = np.array(pd.read_csv("solar_system_census_planets.csv")
                      ["Origin"])
    trainingX, testX, trainingOrigin, testOrigin = \
        data_spliter(X, Origin, 0.5)
    # 2
    probabilities = [probability(trainingX, trainingOrigin, X, testX, zipcode)
                     for zipcode in range(4)]
    # 3
    testProbabilities = np.hstack([elem[1] for elem in probabilities])
    testPrediction = testProbabilities.argmax(axis=1)

    # 4
    correct = (testPrediction == testOrigin).sum()
    print("the fraction of correct predictions :",
          f"{correct} / {testOrigin.size}")

    # 5
    probabilities = np.hstack([elem[0] for elem in probabilities])
    prediction = probabilities.argmax(axis=1)

    plot2d(X, Origin, prediction, mylist)
    plot3d(X, Origin, prediction)

    plt.show()
