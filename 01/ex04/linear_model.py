import pandas as pd
import numpy as np
from mylinearregression import MyLinearRegression as MyLR
import matplotlib.pyplot as plt
from matplotlib import cm


def plot1(x, y, y_hat):
    fig1, ax1 = plt.subplots(num=1)
    ax1.grid(True)
    ax1.plot(x, y_hat, "x--", mew=3,
             c="orange", label="$S_{predict}(pills)$")
    ax1.scatter(x, y, label="$S_{true}(pills)$")
    ax1.set_xlabel("Quantity of blue pills (in micrograms)")
    ax1.set_ylabel("Spacecraft driving score")
    ax1.legend()


def plot2(x, y, model):
    ax = plt.figure(num=2).add_subplot(projection='3d')
    m, n = x.shape
    x1 = np.hstack((np.ones((m, 1)), x))

    thetas = model.thetas
    theta0 = np.linspace(-1, 1, 50) + thetas[0]
    theta1 = np.linspace(-0.2, 0.2, 50) + thetas[1]
    theta0, theta1 = np.meshgrid(theta0, theta1)
    cost = np.subtract(
        np.dot(np.dstack([theta0, theta1]), x1.T), y.reshape(-1))
    cost = (cost * cost).sum(axis=2) / 2 / m
    ax.contour3D(theta0, theta1, cost, 150, cmap='twilight')
    ax.set_xlabel("$\\theta_0$")
    ax.set_ylabel("$\\theta_1$")
    ax.set_zlabel("Cost Function $J(\\theta_0, \\theta_1)$")


if __name__ == "__main__":
    data = pd.read_csv("are_blue_pills_magics.csv")
    Xpill = np.array(data["Micrograms"]).reshape(-1, 1)
    Yscore = np.array(data["Score"]).reshape(-1, 1)

    lm = MyLR(np.array([[89], [-8]]), max_iter=100000)
    lm.fit_(Xpill, Yscore)
    Y_model1 = lm.predict_(Xpill)

    plot1(Xpill, Yscore, Y_model1)
    plot2(Xpill, Yscore, lm)
    print(f'Mean Squared Error: {lm.loss_(Yscore, Y_model1) * 2}')
    plt.show()
