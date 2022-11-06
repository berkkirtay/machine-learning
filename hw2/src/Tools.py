# CSE4088 HW2
# Berk Kırtay
# 150118043

import csv
import random
import numpy as np
import matplotlib.pyplot as plt


def plot(X, Y, W, title):
    X = np.array(X)
    x = X[:, 1]
    y = X[:, 2]
    color = ['blue' if value == 1 else 'red' for value in Y]

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
    plt.title(f'{title} for N={len(X)}')
    plt.legend()
    plt.show()


def generateSample(sample_size, polynomial=None, noise=False):
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
        if noise == True and random.uniform(0, 1) < 0.1:
            Y.append(y * - 1)
        else:
            Y.append(y)

    return X, Y, polynomial


def readData():
    file = open('in.csv')
    csvreader = csv.reader(file)
    rows = []
    for row in csvreader:
        rows.append(row)
    file.close()

    X = []
    y = []
    for row in rows:
        X.append([float(row[0]), float(row[1])])
        y.append(int(row[2]))

    return zip(X, y)


def writeData(X, y):
    file = open('out.csv', 'w+')
    csvwriter = csv.writer(file)
    csvwriter.writerows(zip(X, y))
    file.close()
