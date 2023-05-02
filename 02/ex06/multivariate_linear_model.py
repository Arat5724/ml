import pandas as pd
import numpy as np
from mylinearregression import MyLinearRegression as MyLR
import matplotlib.pyplot as plt

max_iter = 1000000


def plot1(x, y, y_hat, xlabel, ax, i):
    c = [0, 0, 0]
    ax.grid(True)
    c[i] = 0.5
    ax.scatter(x, y, color=c, edgecolor='none', label="Sell price")
    c[i] = 1
    ax.scatter(x, y_hat, s=10, color=c,
               edgecolor='none', label="Prdicted sell price")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("$y$: sell price (in keuros)")
    ax.legend()


def plot2(x, y, thetas, ax):
    m, n = x.shape
    x = np.hstack((np.ones((m, 1)), x))
    theta0 = np.linspace(-15, 15, 50) + thetas[0]
    theta1 = np.linspace(-0.2, 0.2, 50) + thetas[1]
    theta0, theta1 = np.meshgrid(theta0, theta1)
    y_hat = np.dot(np.dstack([theta0, theta1]), x.T)
    y = np.broadcast_to(y.reshape(-1), y_hat.shape)
    cost = ((y_hat - y)**2).sum(axis=2) / 2 / m
    ax.contour3D(theta0, theta1, cost, 150, cmap='twilight')
    ax.set_xlabel("$\\theta_0$")
    ax.set_ylabel("$\\theta_1$")
    ax.set_zlabel("Cost Function $J(\\theta_0, \\theta_1)$")


def uni(data):
    Y = np.array(data["Sell_price"]).reshape(-1, 1)
    fig = plt.figure(figsize=(12, 8))
    myLR_age = MyLR(thetas=np.array(
        [[647], [-13]]), alpha=5e-5, max_iter=max_iter)
    myLR_thrust = MyLR(thetas=np.array(
        [[40], [4]]), alpha=5e-5, max_iter=max_iter)
    myLR_distance = MyLR(thetas=np.array(
        [[745], [-3]]), alpha=5e-5, max_iter=max_iter)

    mylist = [
        ("Age",          myLR_age,      "$x_1$: age (in years)"),
        ("Thrust_power", myLR_thrust,   "$x_2$: thrust power (in 10Km/s)"),
        ("Terameters",   myLR_distance, "$x_3$: distance totalizer value of spacecraft (in 10Km/s)")]
    for i, (feature, myLR, xlabel) in enumerate(mylist):
        X = np.array(data[feature]).reshape(-1, 1)
        myLR.fit_(X, Y)
        Y_model1 = myLR.predict_(X)
        ax1 = fig.add_subplot(2, 3, 1 + i)
        ax2 = fig.add_subplot(2, 3, 4 + i, projection='3d')
        plot1(X, Y, Y_model1, xlabel, ax1, i)
        plot2(X, Y, myLR.thetas, ax2)
        print('Feature:', feature)
        print('\tMean Squared Error:', myLR.loss_(Y, Y_model1) * 2)


def mul(data):
    X = np.array(data[['Age', 'Thrust_power', 'Terameters']])
    Y = np.array(data[['Sell_price']])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    my_lreg = MyLR(thetas=np.array(
        [[385.67], [-24.4], [5.6], [-2.7]]), alpha=1e-5, max_iter=max_iter)
    my_lreg.fit_(X, Y)
    mylist = [
        ("Age",          "$x_1$: age (in years)"),
        ("Thrust_power", "$x_2$: thrust power (in 10Km/s)"),
        ("Terameters",   "$x_3$: distance totalizer value of spacecraft (in Tmeters)")]
    Y_model = my_lreg.predict_(X)
    for i, ((feature, xlabel), ax) in enumerate(zip(mylist, axes)):
        plot1(np.array(data[feature]), Y, Y_model, xlabel, ax, i)
    print('Features: Age, Thrust_power, Terameters')
    print('\tMean Squared Error:', my_lreg.loss_(Y, Y_model) * 2)


if __name__ == "__main__":
    data = pd.read_csv("spacecraft_data.csv")
    uni(data)
    mul(data)
    plt.show()
