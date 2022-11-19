# CSE4088 HW3
# Berk Kırtay
# 150118043

import numpy as np
import random
import matplotlib.pyplot as plt


class Logistic_Regression():
    learning_rate = 0.01

    def __init__(self, d, Y, X):
        self.epochs = 0
        self.d = d + 1
        self.W = np.zeros(d + 1)
        self.X = np.array(X)
        self.Y = np.array(Y)

    # Computing the gradient:

    def stochastic_gradient_descent(self, rand):
        return -self.Y[rand]*self.X[rand] / (1 + np.exp(self.Y[rand] * np.dot(self.W, self.X[rand])))

    def calculate_cross_entropy_error(self):
        err = 0
        for i in range(len(self.Y)):
            err += np.log(1 +
                          np.exp(-self.Y[i] * np.dot(self.W, self.X[i])))
        return err / len(self.Y)

    def logistic_regression_algorithm(self):
        for i in range(1000):
            w_before = self.W.copy()
            # Choose a random permutation from every data point:
            for j in np.random.permutation(len(self.Y)):
                gd = self.stochastic_gradient_descent(j)
                curr_w = self.learning_rate * gd
                # Fixed step size:
                self.W = self.W - curr_w

            self.epochs += 1
            # Calculate difference between weights to decide for halting:
            if np.abs(np.linalg.norm(self.W - w_before)) < 0.01:
                break


def generateSample(sample_size, polynomial=None):
    X = []
    Y = []

    # Choosing a random target function line:
    coefficients = np.polyfit([random.uniform(-1.0, 1.0) for i in range(2)],
                              [random.uniform(-1.0, 1.0) for i in range(2)], 1)
    if polynomial == None:
        polynomial = np.poly1d(coefficients)

    def target_function(x, y):
        return y - polynomial(x)

    for i in range(sample_size):
        x = [1, random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)]
        X.append(x)
        y = np.sign(target_function(X[i][1], X[i][2]))
        Y.append(y)

    return X, Y, polynomial


def plot(X, Y, W):
    X = np.array(X)
    Y = np.array(Y)

    color = ['blue' if value == 1 else 'red' for value in Y]
    x = X[:, 1]
    y = X[:, 2]
    # Scatter points with colors
    plt.scatter(x, y, marker='.', color=color)

    # Solving for (w0 + w1x + w2y = 0) to plot the predicted classification line:
    line = []
    y_i = []

    # And plot the data predicted classification line:
    for i in np.linspace(np.amin(X[:, :2]), np.amax(X[:, :2])):
        # if all the points belong to the same target, one of the
        # weights can be 0, so to avoid division by 0:
        if W[0] == 0:
            W[0] = 0.01
        if W[1] == 0:
            W[1] = 0.01
        if W[2] == 0:
            W[2] = 0.01
        m = -(W[0]/W[2])/(W[0]/W[1])
        b = -W[0]/W[2]

        y = (m*i) + b
        line.append(float(y))
        y_i.append(i)

    plt.plot(y_i, line, '-', label='Predicted Target Function')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.ylabel('Feature y')
    plt.xlabel('Feature x')
    plt.title(f'Logistic Regression Algorithm for N={len(X)}')
    plt.legend()
    plt.show()


def run(is_plot=False):
    average_err = 0
    average_epochs = 0
    for i in range(100):
        X, Y, _ = generateSample(100)
        lr = Logistic_Regression(2, Y, X)
        lr.logistic_regression_algorithm()

        average_err += lr.calculate_cross_entropy_error()
        average_epochs += lr.epochs

    average_err /= 100
    average_epochs /= 100

    print(f"Eout: {average_err}, epochs: {average_epochs}")

    if is_plot == True:
        # Plot the predicted line
        X, Y, _ = generateSample(100)
        lr = Logistic_Regression(2, Y, X)
        lr.logistic_regression_algorithm()
        plot(lr.X, lr.Y, lr.W)


run()
