import pandas as pd
import numpy as np
from mylinearregression import MyLinearRegression as MyLR
import matplotlib.pyplot as plt
from matplotlib import cm


def plot1(x, y, y_hat, xlabel, ax, i):
    c = [0, 0, 0]
    c[i] = 1
    ax.grid(True)
    ax.scatter(x, y_hat, s=10, edgecolor='none',
               color=c, label="Prdicted sell price")
    c[i] = 0.5
    ax.scatter(x, y, color=c, label="Sell price")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("$y$: sell price (in keuros)")
    ax.legend()


def plot2(x, y, thetas, ax, i):
    m, n = x.shape
    x = np.hstack((np.ones((m, 1)), x))

    theta0 = np.linspace(-1, 1, 50) + thetas[0]
    theta1 = np.linspace(-0.2, 0.2, 50) + thetas[1]
    theta0, theta1 = np.meshgrid(theta0, theta1)
    cost = np.subtract(np.dot(np.dstack([theta0, theta1]), x.T), y.reshape(-1))
    cost = (cost * cost).sum(axis=2) / 2 / m
    ax.contour3D(theta0, theta1, cost, 150, cmap='twilight')
    ax.set_xlabel("$\\theta_0$")
    ax.set_ylabel("$\\theta_1$")
    ax.set_zlabel("Cost Function $J(\\theta_0, \\theta_1)$")


def uni(data):
    Y = np.array(data["Sell_price"]).reshape(-1, 1)
    myLR_age = MyLR(np.array([[647], [-13]]), alpha=5e-5, max_iter=100000)
    myLR_thrust = MyLR(np.array([[40], [4]]), alpha=5e-5, max_iter=100000)
    myLR_distance = MyLR(np.array([[745], [-3]]), alpha=5e-5, max_iter=100000)

    mylist = [
        ("Age",          myLR_age,      "$x_1$: age (in years)"),
        ("Thrust_power", myLR_thrust,   "$x_2$: thrust power (in 10Km/s)"),
        ("Terameters",   myLR_distance, "$x_3$: distance totalizer value of spacecraft (in 10Km/s)")]
    for i, (feature, myLR, xlabel) in enumerate(mylist):
        X = np.array(data[feature]).reshape(-1, 1)
        myLR.fit_(X, Y)
        Y_model1 = myLR.predict_(X)
        print(myLR.thetas)
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        plot1(X, Y, Y_model1, xlabel, ax1, i)
        plot2(X, Y, myLR.thetas, ax2, i)
        print(f'Mean Squared Error: {myLR.loss_(Y, Y_model1) * 2}')


def mul(data):
    X = np.array(data[['Age', 'Thrust_power', 'Terameters']])
    Y = np.array(data[['Sell_price']])
    my_lreg = MyLR(theta=np.array(
        [[1.0], [1.0], [1.0], [1.0]]), alpha=1e-4, max_iter=600000)


if __name__ == "__main__":
    data = pd.read_csv("spacecraft_data.csv")
    uni(data)

    plt.show()
