# CSE4088 HW2
# Berk Kırtay
# 150118043

# Requirements:
# Numpy is necessary to run this program.
# For plotting, matplotlib is necessary.

import numpy as np
from Perceptron import *
from Linear_Regression import *
from Tools import *


def run_pla(sample_size, plotting=False, question=""):
    # For questions 4, 5, 6 and 7:
    averageIterations = 0
    averageDisagreements = 0
    for i in range(1000):
        p = Perceptron(0.01, 10000000)
        p.initialize(3, Unit_Step())

        X, Y, _ = generateSample(sample_size)
        p.train(X, Y)
        X, Y, _ = generateSample(sample_size, _)
        averageDisagreements += p.test(X, Y)
        averageIterations += p.iterations

    print(
        f"{question}: For N={sample_size}, Disagreements: {averageDisagreements / 1000}, iterations until convergence: {averageIterations / 1000}.")
    if plotting == True:
        plot(X, Y, p.W, 'Perceptron Learning Algorithm')


def run_linear_regression(sample_size, plotting=False, question=""):
    # For question 8:
    averageDisagreements = 0
    for i in range(1000):
        p = Linear_Regression()
        p.initialize(3, Linear_Regression_Act())

        X, Y, _ = generateSample(sample_size)

        p.train(X, Y)
        averageDisagreements += p.test(X, Y)

    print(
        f"{question}: For N={sample_size}, Ein: {averageDisagreements / 1000}.")
    if plotting == True:
        plot(X, Y, p.W, 'Linear Regression Algorithm')


def run_linear_regression2(sample_size, plotting=False, question=""):
    # For question 9:
    averageDisagreements = 0
    for i in range(1000):
        p = Linear_Regression()
        p.initialize(3, Linear_Regression_Act())

        X, Y, polynomial = generateSample(sample_size)
        p.train(X, Y)

        # Generate out of sample data:
        X_, Y_, _ = generateSample(1000, polynomial)
        averageDisagreements += p.test(X_, Y_)

    print(
        f"{question}: For N={sample_size}, Eout: {averageDisagreements / 1000}.")
    if plotting == True:
        plot(X_, Y_, p.W, 'Linear Regression Algorithm')


def run_lr_pla(sample_size, plotting=False, question=""):
    # For question 10:
    averageIterations = 0
    averageDisagreements = 0
    for i in range(1000):
        l = Linear_Regression()
        l.initialize(3, Linear_Regression_Act())

        X, Y, _ = generateSample(sample_size)
        l.train(X, Y)

        p = Perceptron(0.1, 10)
        p.initialize(3, Unit_Step())
        p.W = l.W
        p.train(X, Y)

        averageDisagreements += p.test(X, Y)
        averageIterations += p.iterations

    print(
        f"{question}: For N={sample_size}, Disagreements: {averageDisagreements / 1000}, iterations: {averageIterations / 1000}.")
    if plotting == True:
        plot(X, Y, p.W, 'Perceptron Learning Algorithm')


def run_nonlinear_transformation(sample_size, plotting=False, question=""):
    # For question 11:
    averageDisagreements = 0

    for i in range(1000):
        p = Linear_Regression()
        p.initialize(3, Nonlinear_Transformation_LR())

        X, Y, _ = generateSample(sample_size, noise=True)

        linear_transformed_X = []
        for x in X:
            linear_transformed_X.append((1, x[0], x[1]))
        p.train(linear_transformed_X, Y)

        averageDisagreements += p.test(linear_transformed_X, Y)

    print(
        f"{question}: For N={sample_size}, Ein: {averageDisagreements / 1000}.")
    if plotting == True:
        plot(X, Y, p.W, 'Linear Regression Algorithm with Nonlinear Transformation')


def run_nonlinear_transformation2(sample_size, plotting=False, question=""):
    # For question 12 and 13:
    averageDisagreements = 0

    for i in range(1000):
        p = Linear_Regression()
        p.initialize(3, Nonlinear_Transformation_LR2())

        X, Y, polynomial = generateSample(sample_size)

        # Linearly transform the data:
        linear_transformed_X = []
        for x in X:
            linear_transformed_X.append((
                1, x[1], x[2], x[1]*x[2], np.power(x[1], 2), np.power(x[2], 2)))
        p.train(linear_transformed_X, Y)

        # Generate out of sample data:
        X, Y, _ = generateSample(sample_size, polynomial)

        linear_transformed_X = []
        for x in X:
            linear_transformed_X.append([
                1, x[1], x[2], x[1]*x[2], np.power(x[1], 2), np.power(x[2], 2)])

        averageDisagreements += p.test(linear_transformed_X, Y) / 5

    print(
        f"{question}: For N={sample_size}, Eout: {averageDisagreements / 1000}")
    if plotting == True:
        plot(X, Y, p.W,
             'Linear Regression Algorithm with Nonlinear Transformation')


'''
To activate plotting, call the functions with True flag:
"run_pla(sample_size=10, plotting=True, question="Q4, Q5")"
'''

run_pla(sample_size=10, plotting=False, question="Q4, Q5")
run_pla(sample_size=100, plotting=False, question="Q6, Q7")
run_linear_regression(sample_size=100, plotting=False, question="Q8")
run_linear_regression2(sample_size=100, plotting=False, question="Q9")
run_lr_pla(sample_size=10, plotting=False, question="Q10")
run_nonlinear_transformation(sample_size=1000, plotting=False, question="Q11")
run_nonlinear_transformation2(
    sample_size=1000, plotting=False, question="Q12, Q13")
