# CSE4088 HW2
# Berk Kırtay
# 150118043

# This file contains the activation function methods for PLA and linear regression etc..

import numpy as np


class Activation():
    def activation_func(self, res):
        return res

    def test(self, X, W):
        return 0


class Unit_Step(Activation):
    def activation_func(self, res):
        return self.unit_step(res)

    def unit_step(self, x):
        return np.sign(x)

    def test(self, X, W):
        return np.dot(X, W)


class Linear_Regression_Act(Activation):
    def activation_func(self, res):
        return self.linear_regression(res)

    def linear_regression(self, x):
        return np.sign(x)

    def test(self, X, W):
        return np.dot(X, W)


class Nonlinear_Transformation_LR(Activation):
    def activation_func(self, res):
        return self.linear_regression(res)

    def linear_regression(self, x):
        return np.sign(x)

    def test(self, X, W):
        return np.power(X[1], 2) + np.power(X[2], 2) - 0.6


class Nonlinear_Transformation_LR2(Activation):
    def activation_func(self, res):
        return self.linear_regression(res)

    def linear_regression(self, x):
        return np.sign(x)

    def test(self, X, W):
        return -1 - 0.05*float(X[1]) + 0.08*float(X[2]) + 0.13*float(X[1]) * float(X[2]) + 1.5 * np.power(X[1], 2) + 15 * np.power(X[2], 2)
