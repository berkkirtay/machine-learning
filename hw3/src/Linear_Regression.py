# CSE4088 HW3
# Berk Kırtay
# 150118043

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Linear_Regression():
    activation = None

    def __init__(self, size):
        self.disagreements = 0
        self.iterations = 0
        self.W = np.zeros(size)

    def train(self, X, Y):
        x = np.matrix(X)
        y = np.matrix(Y)

        # Apply Linear Regression Algorithm:
        pseudo_inverse = np.dot(np.linalg.pinv(np.dot(x.T, x)), x.T)
        self.W = np.dot(pseudo_inverse, y.T)

    def train_with_decay(self, X, Y, k):
        x = np.matrix(X)
        y = np.matrix(Y)

        # Apply Linear Regression Algorithm with adding decay 10^k:
        pseudo_inverse = np.dot(np.linalg.pinv(
            np.dot(x.T, x) + np.identity(len(self.W)) * 10 ** k), x.T)
        self.W = np.dot(pseudo_inverse, y.T)

    def predict(self, X, W):
        return np.sign(np.dot(X, W))

    def test(self, X, Y):
        disagreements = 0
        predicted_Y = []
        for i in range(len(X)):
            res = self.predict(X[i], self.W)
            disagreements += res != Y[i]
            predicted_Y.append(res)
        return float(disagreements / len(X)), predicted_Y


def get_train_data():
    in_dtafile = './in.dta'
    data = np.loadtxt(in_dtafile)
    return data[:, :2], data[:, 2]


def get_test_data():
    out_dtafile = './out.dta'
    data = np.loadtxt(out_dtafile)
    return data[:, :2], data[:, 2]


def plot(X, Y):
    X = np.array(X)
    Y = np.array(Y)

    color = ['blue' if value == 1 else 'red' for value in Y]
    x = X[:, 0]
    y = X[:, 1]
    # Scatter points with colors
    plt.scatter(x, y, marker='.', color=color)

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.ylabel('Feature y')
    plt.xlabel('Feature x')
    plt.title(f'Linear Regression Algorithm for N={len(X)}')
    plt.show()


def non_linear_transformation(X):
    # φ(x1, x2) = (1, x1, x2, x1^2, x2^2, x1*x2, | x1 − x2 |, | x1 + x2|).
    non_linear_X = []
    for x in X:
        non_linear_X.append([1, x[0], x[1], x[0]**2, x[1]**2, x[0]
                             * x[1], np.abs(x[0] - x[1]), np.abs(x[0] + x[1])])
    return np.array(non_linear_X)


def run_q2(is_plot=False):
    X_in, Y_in = get_train_data()
    non_linear_X_in = non_linear_transformation(X_in)

    lr = Linear_Regression(len(non_linear_X_in[0]))
    lr.train(non_linear_X_in, Y_in)

    X_out, Y_out = get_test_data()
    non_linear_X_out = non_linear_transformation(X_out)

    # Test in-sample and out-of-sample classification errors:
    err_in, predicted_Y_in = lr.test(non_linear_X_in, Y_in)
    err_out, predicted_Y_out = lr.test(non_linear_X_out, Y_out)

    print(f'Q2: Ein: {err_in}, Eout: {err_out}')

    if is_plot == True:
        # Scatter real and predicted points for training data:
        plot(X_in, Y_in)
        plot(X_in, predicted_Y_in)

        # Scatter real and predicted points for test data:
        plot(X_out, Y_out)
        plot(X_out, predicted_Y_out)


def run_q3():
    X_in, Y_in = get_train_data()
    non_linear_X_in = non_linear_transformation(X_in)

    lr = Linear_Regression(len(non_linear_X_in[0]))
    lr.train_with_decay(non_linear_X_in, Y_in, -3)

    X_out, Y_out = get_test_data()
    non_linear_X_out = non_linear_transformation(X_out)

    # Test in-sample and out-of-sample classification errors:
    err_in, predicted_Y_in = lr.test(non_linear_X_in, Y_in)
    err_out, predicted_Y_out = lr.test(non_linear_X_out, Y_out)

    print(f'Q3: Ein: {err_in}, Eout: {err_out}')


def run_q4():
    X_in, Y_in = get_train_data()
    non_linear_X_in = non_linear_transformation(X_in)

    lr = Linear_Regression(len(non_linear_X_in[0]))
    lr.train_with_decay(non_linear_X_in, Y_in, 3)

    X_out, Y_out = get_test_data()
    non_linear_X_out = non_linear_transformation(X_out)

    # Test in-sample and out-of-sample classification errors:
    err_in, predicted_Y_in = lr.test(non_linear_X_in, Y_in)
    err_out, predicted_Y_out = lr.test(non_linear_X_out, Y_out)

    print(f'Q4: Ein: {err_in}, Eout: {err_out}')


def run_q5_q6():
    X_in, Y_in = get_train_data()
    non_linear_X_in = non_linear_transformation(X_in)
    X_out, Y_out = get_test_data()
    non_linear_X_out = non_linear_transformation(X_out)

    smallest_err_out = 1
    smallest_k = 0

    # Trying the values between -10 and 10 for k to get the smallest Eout possible:
    for k in range(-10, 10):
        lr = Linear_Regression(len(non_linear_X_in[0]))
        lr.train_with_decay(non_linear_X_in, Y_in, k)

        # Test in-sample and out-of-sample classification errors:
        err_in, predicted_Y_in = lr.test(non_linear_X_in, Y_in)
        err_out, predicted_Y_out = lr.test(non_linear_X_out, Y_out)

        if err_out < smallest_err_out:
            smallest_err_out = err_out
            smallest_k = k

    print(f'Q5- Q6: k: {smallest_k}, Eout: {smallest_err_out}')


run_q2(True)
run_q3()
run_q4()
run_q5_q6()
