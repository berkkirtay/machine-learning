# CSE4088 HW3
# Berk Kırtay
# 150118043

import random
import numpy as np
import matplotlib.pyplot as plt


class Perceptron():
    def __init__(self, size):
        self.W = [0.1 for i in range(size)]
        self.W = np.array(self.W)
        self.X = 0
        self.S = 0


class NeuralNetwork():
    def __init__(self, size, layers):
        self.size = size
        self.last_s = 0
        self.learning_rate = 0.01
        self.totalOperations = 0

        # Layers representation:
        # [input layer, hidden layer(s), output layer]

        # Initialize neural network:
        self.perceptrons = []

        # Creates a fully connected neural network:
        last_d = layers[0]
        for i in range(1, size + 1):
            self.perceptrons.append([Perceptron(last_d)
                                    for j in range(layers[i])])
            last_d = layers[i]

    # We use tanh function as activation:
    def activation_function(self, s):
        # => (np.exp(s) - np.exp(-s)) / (np.exp(s) + np.exp(-s))
        return np.tanh(s)

    def stochastic_gradient_descent(self, index):
        pass

    def forward_propagation(self, x):
        x = np.array(x)

        # Calculate input of every perceptron:
        for i in range(self.size):
            for j in range(len(self.perceptrons[i])):
                s_ij = np.dot(self.perceptrons[i]
                              [j].W, x)
                self.perceptrons[i][j].X = self.activation_function(s_ij)
                if i == self.size - 1:
                    self.last_s = s_ij
                self.totalOperations += x.size

            # initialize the next inputs:
            x = np.array([x.X for x in self.perceptrons[i]])

        # Return the predicted X:
        res = 0
        for i in self.perceptrons[-1]:
            res += i.X
        return res

    def backward_propagation(self, y):
        # Calculate final layer Sj(L) = 2(x - y)(1 - tanh^2(s)):
        x_L = self.perceptrons[-1][-1].X
        theta = self.activation_function(self.last_s)
        S_L = 2 * (x_L - y) * (1 - theta ** 2)
        self.perceptrons[-1][-1].S = S_L
        self.totalOperations += 1

        # Calculate the rest of S_i:
        for i in range(len(self.perceptrons) - 2, -1, -1):
            for j in range(len(self.perceptrons[i])):
                cross_w_s = 0
                for x in range(len(self.perceptrons[i + 1])):
                    cross_w_s += self.perceptrons[i +
                                                  1][x].W[j] * self.perceptrons[i + 1][x].S
                    self.totalOperations += 1

                self.perceptrons[i][j].S = (
                    1 - self.perceptrons[i][j].X ** 2) * cross_w_s
                self.totalOperations += 1

        # Update the weights:
        for i in range(len(self.perceptrons) - 2, -1, -1):
            for j in range(len(self.perceptrons[i])):
                for x in range(len(self.perceptrons[i + 1])):
                    update = self.learning_rate * \
                        self.perceptrons[i][j].X * self.perceptrons[i + 1][x].S
                    self.perceptrons[i + 1][x].W[j] -= update
                    self.totalOperations += 1

    def train(self, X, Y):
        for i in range(1):
            for j in range(len(Y)):
                self.forward_propagation(X[j])
                self.backward_propagation(Y[j])

    def test(self, X, Y):
        err = 0
        for i in range(len(Y)):
            predicted_y = self.forward_propagation(X[i])
            err += 0.5 < np.abs(predicted_y - Y[i])

        return err / len(Y)


def test_neural_network():
    X = [[1, 2, 3, 4, 2], [1, 5, 3, 4, 2], [
        1, 2, 3, 4, 12], [-1, 2, -3, 4, 2]]

    Y = [0.1, 0.2, 1, 0.01]

    nn = NeuralNetwork(2, [5, 3, 1])
    for i in range(100):
        nn.train(X, Y)

    print(f'Ein: {nn.test(X, Y)}')


def run_q8():
    X = [[1, 2, 3, 4, 2]]
    Y = [1]
    nn = NeuralNetwork(2, [5, 3, 1])

    nn.train(X, Y)

    print(f'Q8: Total number of operations is: {nn.totalOperations}')


def run_q9():
    X = []
    Y = []
    for i in range(1):
        temp = []
        for j in range(10):
            temp.append(random.uniform(-100, 100))
        X.append(temp)
        Y.append(random.randint(0, 1))

    # Here we set 36 hidden layers with 1
    # weight to get the minimum value of weights:
    layers = [1 for i in range(36)]

    # Adding first layer:
    layers.insert(0, 10)

    # Adding final layer:
    layers.append(1)

    nn = NeuralNetwork(len(layers) - 1, layers)

    nn.train(X, Y)

    # Counting weights:
    weights = 0
    for layer in nn.perceptrons:
        for perceptron in layer:
            weights += perceptron.W.size

    print(f'Q9: Minimum number of weights is: {weights}')


def run_q10():
    X = []
    Y = []
    for i in range(1):
        temp = []
        for j in range(10):
            temp.append(random.uniform(-100, 100))
        X.append(temp)
        Y.append(random.randint(0, 1))

    # Here we set hidden layers like in the following:
    layers = [18, 18]

    # Adding first layer:
    layers.insert(0, 10)

    # Adding final layer:
    layers.append(1)

    nn = NeuralNetwork(len(layers) - 1, layers)

    # Counting weights:
    weights = 0
    for layer in nn.perceptrons:
        for perceptron in layer:
            weights += perceptron.W.size

    print(f'Q10: Maximum number of weights is: {weights}')


run_q8()
run_q9()
run_q10()
