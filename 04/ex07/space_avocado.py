import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import Series
from ridge import MyRidge
import matplotlib.pyplot as plt
import pickle


def add_polynomial_features(x: ndarray, power: int):
    x = x.reshape(-1)
    v = np.empty((power, x.shape[0]), dtype=x.dtype)
    v[0] = x
    for i in range(1, power):
        v[i] = v[i - 1] * x
    return v.T


def minmax(x: ndarray) -> ndarray:
    xmin, xmax = x.min(), x.max()
    return (x - xmin)/(xmax - xmin)


def scaler(x: Series, n: int) -> ndarray:
    return add_polynomial_features(minmax(x.to_numpy()), n)


if __name__ == "__main__":
    with open("models.pickle", "rb") as f:
        mydict = pickle.load(f)
    mydict = sorted(
        mydict.items(), key=lambda item: item[1]['mse'], reverse=True)

    # Plot the evaluation curve which help you to select the best model #
    models_abc = [f"{v[0]} {v[1]['lambda_']}" for v in mydict]
    models_mse = [v[1]['mse'] for v in mydict]
    ax1 = plt.figure(figsize=(15, 5)).add_subplot()
    ax1.plot(models_abc, models_mse)
    plt.xticks(rotation=90)

    # select the best model #
    best_model = mydict[-1]
    key, lambda_, thetas = best_model[0], best_model[1]['lambda_'], best_model[1]['theta']
    mylr = MyRidge(thetas=thetas, lambda_=lambda_)

    # read a data #
    data = pd.read_csv("space_avocado.csv")
    mylist = ["weight", "prod_distance", "time_delivery"]
    X = np.hstack(tuple(scaler(data[feature], int(n))
                        for feature, n in zip(mylist, str(key))))
    Y = data["target"].to_numpy()
    Y_model = mylr.predict_(X)

    # Plot the true price and the predicted price obtain via your best model #
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for feature, ax in zip(mylist, axes):
        Xi = data[feature].to_numpy()
        ax.scatter(Xi, Y_model, s=1, color=(0, 0, 1), label="predicted")
        ax.scatter(Xi, Y, s=1, color=(1, 0, 0), label="real")
        ax.set_xlabel(feature)
        ax.set_ylabel("target")
        ax.legend()
    plt.show()
