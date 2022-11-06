# CSE4088 HW2
# Berk Kırtay
# 150118043

import numpy as np
from Activation import *


class Linear_Regression():
    activation = None

    def __init__(self):
        self.disagreements = 0
        self.iterations = 0

    def initialize(self, size: int, activation: Activation):
        self.W = np.zeros(size)
        self.activation = activation

    def train(self, X, Y):
        x = np.matrix(X)
        y = np.matrix(Y)

        # Apply Linear Regression Algorithm:
        pseudo_inverse = np.dot(np.linalg.pinv(np.dot(x.T, x)), x.T)
        self.W = np.dot(pseudo_inverse, y.T)

    def test(self, X, Y):
        disagreements = 0
        for i in range(len(X)):
            res = self.activation.test(X[i], self.W)
            disagreements += self.activation.activation_func(res) != Y[i]
        return float(disagreements / len(X))
