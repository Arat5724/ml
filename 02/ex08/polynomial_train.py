import numpy as np
from numpy import ndarray
import pandas as pd
from mylinearregression import MyLinearRegression as MyLR
import matplotlib.pyplot as plt


def add_polynomial_features(x: ndarray, power: int) -> ndarray | None:
    x = x.reshape(-1)
    v = np.empty((power, x.shape[0]), dtype=x.dtype)
    v[0] = x
    for i in range(1, power):
        v[i] = v[i - 1] * x
    return v.T


if __name__ == "__main__":
    fig, ax = plt.subplots()
    ax2 = plt.figure().add_subplot()
    data = pd.read_csv("are_blue_pills_magics.csv")
    X0 = np.array(data["Micrograms"])
    X = add_polynomial_features(X0, 6)
    Xp0 = np.linspace(X0.min(), X0.max(), 100)
    Xp = add_polynomial_features(Xp0, 6)
    Y = np.array(data["Score"]).reshape(-1, 1)
    thetas = [
        ([[89], [-9]], 1e-4),
        ([[92], [-11], [0]], 1e-5),
        ([[84], [-1], [-3], [0]], 1e-6),
        ([[-20], [160], [-78], [14], [-1]], 1e-7),
        ([[1140], [-1850], [1110], [-305], [40], [-2]], 1e-8),
        ([[9110], [-18015], [13400], [-4935], [966], [-96.4], [3.86]], 1e-9)
    ]
    mse_x = [1, 2, 3, 4, 5, 6]
    mse_height = [None] * 6
    ax.scatter(X0, Y)
    for i, (theta, alpha) in enumerate(thetas, 1):
        mylr = MyLR(np.array(theta).reshape(-1, 1), alpha, 100000)
        Xi = X[..., :i]
        mylr.fit_(Xi, Y)
        Xt = Xp[..., :i]
        Yp0 = mylr.predict_(Xt)
        y_hat = mylr.predict_(Xi)
        mse = mylr.loss_(Y, y_hat) * 2
        print(f"Degree: {i}")
        print(f"\tThetas: {mylr.thetas.tolist()}")
        print(f"\tMSE: {mse}")
        mse_height[i - 1] = mse
        ax.plot(Xp0, Yp0, label=f"{i}")
    fig.legend()
    ax2.bar(mse_x, mse_height)
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("MSE")
    plt.show()
