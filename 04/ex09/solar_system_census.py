import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
from my_logistic_regression import MyLogisticRegression as MyLR
import matplotlib.pyplot as plt
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


def get_f1_score_(models, X, Y):
    Y_pred = np.hstack(
        [model.predict_(X) for model in models]).argmax(axis=1).reshape((-1, 1))
    return multiclass_f1_score_(Y, Y_pred)


def confusion_matrix_(y_true: ndarray, y_hat: ndarray,
                      labels: list | None = None, df_option: bool = False) -> ndarray | DataFrame | None:
    y_true, y_hat = y_true.reshape(-1), y_hat.reshape(-1)
    if labels is None:
        labels = list(set(y_true.tolist() + y_hat.tolist()))
        labels.sort()
    true = np.vstack([y_true == label for label in labels], dtype=int)
    prediction = np.vstack([y_hat == label for label in labels], dtype=int)
    if df_option:
        return DataFrame(data=np.dot(true, prediction.T), index=labels, columns=labels)
    return np.dot(true, prediction.T)


if __name__ == "__main__":
    with open("models.pickle", "rb") as f:
        mydict = pickle.load(f)

    # performance
    perfAx = plt.figure(num="Performance").add_subplot()
    sortedMydict = sorted(
        mydict.items(), key=lambda x: x[1]['f1'], reverse=True)
    perfx = [str(ele[0]) for ele in sortedMydict]  # lambda_
    perfy = [ele[1]['f1'] for ele in sortedMydict]  # f1_score_
    perfAx.bar(perfx, perfy)

    mylist = ["weight", "height", "bone_density"]
    X0 = pd.read_csv("solar_system_census.csv")
    X = X0.copy()
    for feature in mylist:
        X[feature] = minmax(X[feature])

    X0, X = np.array(X0[mylist]), np.array(X[mylist])
    X = add_polynomial_features(X, 3)
    Origin = np.array(pd.read_csv("solar_system_census_planets.csv")
                      ["Origin"]).reshape((-1, 1))
    trainingX, testX, trainingOrigin,  testOrigin = (
        X[:60], X[100:], Origin[:60], Origin[100:])

    # f1 score on test set
    for k, v in mydict.items():
        print(f"lambda_: {k}")
        print(f"\tf1 score: {get_f1_score_(v['models'], testX, testOrigin)}")

    # Best model
    theta, max_iter, lambda_ = (
        np.array([[0]] * 10), 1000000, sortedMydict[0][0])
    bestModel = [MyLR(theta, max_iter=max_iter, lambda_=lambda_)
                 for _ in range(4)]
    for zipcode, mylr in enumerate(bestModel):
        trainingY = (trainingOrigin == zipcode) * 1
        mylr.fit_(trainingX, trainingY)
    Origin_pred = np.hstack(
        [mylr.predict_(X) for mylr in bestModel]).argmax(axis=1).reshape((-1, 1))
    trueAx = plt.figure(num="true").add_subplot(projection="3d")
    predAx = plt.figure(num="prediction").add_subplot(projection="3d")
    mixedAx = plt.figure(num="mixed").add_subplot(projection="3d")
    xT = X0.T
    color = ['yellow', 'blue', 'red', 'black']
    trueColor = [color[int(area[0])] for area in Origin]
    trueAx.scatter(xT[0], xT[1], xT[2], c=trueColor)
    predColor = [color[int(area[0])] for area in Origin_pred]
    predAx.scatter(xT[0], xT[1], xT[2], c=predColor)
    mixedColor = [c1 if c1 == c2 else 'green'
                  for c1, c2 in zip(trueColor, predColor)]
    mixedAx.scatter(xT[0], xT[1], xT[2], c=mixedColor)

    print(confusion_matrix_(Origin, Origin_pred, df_option=True))
    plt.show()
