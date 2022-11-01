from ssl import CHANNEL_BINDING_TYPES
import numpy as np
import random
import csv
import matplotlib.pyplot as plt


class Perceptron():
    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.maxIterations = iterations
        self.predictions = []
        self.disagreements = 0
        self.iterations = 0
        self.minErr = 1

    def initialize(self, size):
        self.W = np.zeros(size)
        self.bias = 0

    def activation_func(self, res):
        return self.unit_step(res)

    def unit_step(self, x):
        return np.where(x >= 0, 1, 0)

    def train(self, dataset):
        repeats = 0
        for i in range(self.maxIterations):
            err = self.bias
            for x, y in dataset:
                repeats += 1
                res = np.cross(x, self.W) + self.bias
                self.predictions.append(res)

                predicted = self.activation_func(res)

                err = y - predicted

                change = err * self.learning_rate

                self.bias += change

                for i in range(len(self.W)):
                    self.W[i] += change * x[i]

            if abs(err) < self.minErr:
                self.iterations = repeats
                self.minErr = abs(err)

    def test(self, X):
        res = np.cross(X, self.W) + self.bias
        return self.activation_func(res)


def run(sample_size):
    p = Perceptron(0.01, 1000)
    p.initialize(2)

    X = []
    Y = []
    res = []
    predictions = []

    coefficients = np.polyfit([random.uniform(-1.0, 1.0) for i in range(2)],
                              [random.uniform(-1.0, 1.0) for i in range(2)], 1)
    polynomial = np.poly1d(coefficients)

    def target_function(x, y):
        return y - polynomial(x)

    for i in range(sample_size):
        X.append([random.uniform(-1.0, 1.0) for i in range(2)])
        Y.append(np.where(target_function(
            X[i][0], X[i][1]) >= 0, 1, 0))

    trainnig_data = zip(X, Y)

    p.train(trainnig_data)

    predictions = np.cross(X, p.W) + p.bias
    predictions = list(predictions)
    predictions.append(1)
    predictions.insert(0, -1)

    disagreements = 0
    for i in range(len(X)):
        res = p.test(X[i])
        if res != Y[i]:
            disagreements += 1

    print(
        f"Disagreements: {disagreements / len(X)}, iterations: {p.iterations}")
    plot(X, Y, predictions, polynomial)


def plot(X, Y, predictions, polynomial):
    X = np.array(X)
    x = X[:, 0]
    y = X[:, 1]
    color = ['blue' if value == 1 else 'red' for value in Y]

    plt.scatter(x, y, marker='.', color=color)
    plt.plot(predictions, polynomial(predictions), '-')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    plt.show()


'''
    file = open('data.csv')
    csvreader = csv.reader(file)
    rows = []
    for row in csvreader:
        rows.append(row)
    file.close()

    for row in rows:
        X.append([float(row[0]), float(row[1])])
        y.append(int(row[2]))

    for i in range(len(rows)):
        p.train(X[i], y[i])

   # for i in range(1000):
    #    res.append(p.test(X[i]))
'''

run(10)
run(100)
