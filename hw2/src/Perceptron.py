
import numpy as np
from Activation import *


class Perceptron():
    activation = None

    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.maxIterations = iterations
        self.iterations = 0

    def initialize(self, size: int, activation: Activation):
        self.W = np.zeros(size)
        self.activation = activation

    def activation_func(self, res):
        return self.activation.activation_func(res)

    def train(self, X, Y):
        for i in range(self.maxIterations):
            convergence = 0
            self.iterations += 1
            misclassified = []
            for j in range(len(X)):
                res = np.dot(X[j], self.W)
                predicted = self.activation_func(res)
                err = Y[j] != predicted
                if err == True:
                    convergence += 1
                    misclassified.append(j)

            if convergence == 0:
                break

            # Pick a random misclassified point to apply PLA with it:
            rand = np.random.choice(misclassified)
            change = Y[rand] * self.learning_rate

            for j in range(len(self.W)):
                self.W[j] += change * X[rand][j]

    def test(self, X, Y) -> float:
        disagreements = 0
        for i in range(len(X)):
            res = self.activation.test(X[i], self.W)
            disagreements += self.activation.activation_func(res) != Y[i]
        return float(disagreements / len(X))
