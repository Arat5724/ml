import pandas as pd
import numpy as np
from numpy import ndarray
from mylinearregression import MyLinearRegression as MyLR
import matplotlib.pyplot as plt
from matplotlib import cm


def add_polynomial_features(x: ndarray, power: int):
    # power must greater than or equal to 0
    x = x.reshape(-1)
    (m,) = x.shape
    v = np.empty((power, m), dtype=x.dtype)
    v[0] = x
    for i in range(1, power):
        v[i] = v[i - 1] * x
    return np.transpose(v)


if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot()

    data = pd.read_csv("are_blue_pills_magics.csv")
    X0 = np.array(data["Micrograms"])
    X = add_polynomial_features(X0, 6)
    Xp0 = np.linspace(X0.min(), X0.max(), 100)
    Xp = add_polynomial_features(Xp0, 6)
    Y = np.array(data["Score"]).reshape(-1, 1)
    thetas = [
        ([[89], [-9]], 1e-4),
        ([[92], [-11], [0]], 1e-5),
        ([[52], [32], [-13], [1]], 1e-6),
        ([[-20], [160], [-80], [10], [-1]], 1e-7),
        ([[1140], [-1850], [1110], [-305], [40], [-2]], 1e-8),
        ([[9110], [-18015], [13400], [-4935], [966], [-96.4], [3.86]], 1e-9)
    ]
    ax.scatter(X0, Y)
    for i, (theta, alpha) in enumerate(thetas, 1):
        mylr = MyLR(np.array(theta).reshape(-1, 1), alpha, 10000000)
        Xi = X[..., :i]
        print(Xi.shape)
        mylr.fit_(Xi, Y)
        print(mylr.thetas)
        Xt = Xp[..., :i]
        Y_model = mylr.predict_(Xt)
        ax.plot(Xp0, Y_model, label=f"{i}")
    fig.legend()
    plt.show()
