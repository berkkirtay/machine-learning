# CSE4088 HW4
# Berk Kirtay
# 150118043

from sklearn import svm
import numpy as np
import random


def readData(path):
    data = np.loadtxt(path)
    X = data[:, 1:]  # Second and third column
    Y = data[:, 0]  # First column
    return np.array(X), np.array(Y)


x_train, y_train = readData('features.train')
x_test, y_test = readData('features.test')


def transform_to_1_versus_all(data, classifier):
    # Transforms the classifier data by 1 and -1:
    y = []
    for c in data:
        curr_classifier = 0
        if c == classifier:
            curr_classifier = 1
        else:
            curr_classifier = -1
        y.append(curr_classifier)
    return np.array(y)


def transform_to_1_versus_1(data_x, data_y, classifiers):

    # Transforms the classifier data by 1 and -1 by
    # including only the given classifiers:
    x = []
    y = []
    for i in range(len(data_y)):
        if data_y[i] == classifiers[0] or data_y[i] == classifiers[1]:
            y.append(data_y[i])
            x.append(data_x[i])

    return np.array(x), np.array(y)

# SVM runner for q1, q2 and q3:


def SVM_run(kernel_type, classifiers, C=0.01, degree=2):
    errors_Ein = []
    errors_Eout = []
    model = None
    for c in classifiers:
        # Transforming the data:
        temp_y_train = transform_to_1_versus_all(y_train, c)
        temp_y_test = transform_to_1_versus_all(y_test, c)

        # Initialization of SVM:
        # (1 + x_n^Tx_m)^Q, here 1 is coef0, so we set @coef0 as 1:
        #  @Degree=Q, @kernel_type=polynomial and @gamma is a effect rate
        # for data fitting.
        model = svm.SVC(C=C, degree=degree, kernel=kernel_type,
                        coef0=1, gamma=1)
        # Training:
        model.fit(x_train, temp_y_train)
        # Testing:
        result = model.score(x_train, temp_y_train)
        # Setting Ein and Eout:
        Ein = (1 - result) * 100  # Ein = 1 - accuracy
        errors_Ein.append(Ein)

        result = model.score(x_test, temp_y_test)
        Eout = (1 - result) * 100  # Eout = 1 - accuracy
        errors_Eout.append(Eout)

    return errors_Eout, errors_Ein, model


def q2():
    classifiers = [0, 2, 4, 6, 8]
    _, errors, _ = SVM_run("poly", classifiers)
    res = np.argmax(errors)

    print(
        f"Q2) The classifier {classifiers[res]} has the highest Ein among others with Ein: {max(errors)}%\n")


def q3():
    classifiers = [1, 3, 5, 7, 9]
    _, errors, _ = SVM_run("poly", classifiers)
    res = np.argmin(errors)
    print(
        f"Q3) The classifier {classifiers[res]} has the lowest Ein among others with Ein: {min(errors)}%\n")


def q4():
    classifiers = [0]
    _, _, model_0 = SVM_run("poly", classifiers)

    classifiers = [1]
    _, _, model_1 = SVM_run("poly", classifiers)

    # @len(model_0.support_vectors_) returns the number of support vectors for each class
    res = len(model_0.support_vectors_) - len(model_1.support_vectors_)
    print(
        f"Q4) Difference between the number of support vectors of these two classifiers is {res}\n")


def q5_q6():
    # For Q5, answer d must be correct because the maximum C
    # achieves the lowest Ein and all other values do not strictly
    # decrease as it is mentioned in the other answers.
    # For Q6, The number of support vectors is lower at Q=5 for C = 0.001.

    # Given C values:
    Cs = [0.001, 0.01, 0.1, 1]
    classifiers = [1, 5]
    # Transforming both train and test data:
    temp_x_train, temp_y_train = transform_to_1_versus_1(
        x_train, y_train, classifiers)
    temp_x_test, temp_y_test = transform_to_1_versus_1(
        x_test, y_test, classifiers)

    for C in Cs:
        model = svm.SVC(C=C, degree=2, kernel="poly",
                        coef0=1, gamma=1)
        model.fit(temp_x_train, temp_y_train)
        result = model.score(temp_x_train, temp_y_train)
        Ein = (1 - result) * 100

        result = model.score(temp_x_test, temp_y_test)
        Eout = (1 - result) * 100
        print(
            f"Q5) For C={C}: Ein={Ein}%, Eout={Eout}, Number of support vectors:{len(model.support_vectors_)}")

    print("\n")

    Cs = [0.0001, 0.001, 0.01, 0.1, 1]
    Qs = [2, 5]
    for Q in Qs:
        for C in Cs:
            model = svm.SVC(C=C, degree=Q, kernel="poly",
                            coef0=1, gamma=1)
            model.fit(temp_x_train, temp_y_train)
            result = model.score(temp_x_train, temp_y_train)
            Ein = (1 - result) * 100

            result = model.score(temp_x_test, temp_y_test)
            Eout = (1 - result) * 100
            print(
                f"Q6) For C={C} and Q={Q}: Ein={Ein}%, Eout={Eout}, Number of support vectors:{len(model.support_vectors_)}")
    print("\n")


def q7():
    Cs = [0.0001, 0.001, 0.01, 0.1, 1]
    classifiers = [1, 5]
    temp_x_train, temp_y_train = transform_to_1_versus_1(
        x_train, y_train, classifiers)
    size = len(temp_x_train)
    selectedOnes = {0.0001: 0, 0.001: 0, 0.01: 0, 0.1: 0, 1: 0}

    for i in range(100):
        lowestEin = 100
        bestC = 0
        for C in Cs:
            # Generating a random number between the possible indexes:
            rand = random.randint(int(size / 10), int(9 * size / 10))
            # Cutting the data array into half by generated random value:
            test_X = temp_x_train[rand:rand + int(size / 10)]
            test_Y = temp_y_train[rand:rand + int(size / 10)]
            model = svm.SVC(C=C, degree=2, kernel="poly",
                            coef0=1, gamma=1)
            # Then we use 90% of the training data for training and
            # use the remaining 10% for test:
            model.fit(temp_x_train, temp_y_train)
            result = model.score(test_X, test_Y)
            Ein = (1 - result) * 100
            # Keeping track of Ein with lowest value:
            if Ein < lowestEin:
                lowestEin = Ein
                bestC = C
        # Then we increment the count of the C value with lowest Ein:
        selectedOnes[bestC] += 1

    print("Q7) The selection numbers of C's:")
    print(selectedOnes)
    print("\n")


def q8():
    # For Q8, we calculate the mean Ecv value after 100 runs.
    C = 0.001
    classifiers = [1, 5]
    temp_x_train, temp_y_train = transform_to_1_versus_1(
        x_train, y_train, classifiers)
    size = len(temp_x_train)

    meanEcv = 0
    for i in range(100):
        rand = random.randint(int(size / 10), int(9 * size / 10))
        test_X = temp_x_train[rand:rand + int(size / 10)]
        test_Y = temp_y_train[rand:rand + int(size / 10)]
        model = svm.SVC(C=C, degree=2, kernel="poly",
                        coef0=1, gamma=1)
        model.fit(temp_x_train, temp_y_train)
        result = model.score(test_X, test_Y)
        Ein = (1 - result)
        meanEcv += Ein
    meanEcv /= 100

    print(f"Q8) The average value of Ecv over the 100 runs is {meanEcv}\n")


def q9_q10():
    # Here we calculate lowest Ein and Eout counts for
    # every C value and print them out:
    classifiers = [1, 5]
    temp_x_train, temp_y_train = transform_to_1_versus_1(
        x_train, y_train, classifiers)
    temp_x_test, temp_y_test = transform_to_1_versus_1(
        x_test, y_test, classifiers)

    lowestEin = 100
    C_Ein = 0
    lowestEout = 100
    C_Eout = 0
    Cs = [0.01, 1, 100, 10**4, 10**6]

    for C in Cs:
        model = svm.SVC(C=C, degree=2, kernel="rbf",
                        coef0=1, gamma=1)
        model.fit(temp_x_train, temp_y_train)
        result = model.score(temp_x_train, temp_y_train)
        Ein = (1 - result) * 100
        if Ein < lowestEin:
            lowestEin = Ein
            C_Ein = C

        result = model.score(temp_x_test, temp_y_test)
        Eout = (1 - result) * 100
        if Eout < lowestEout:
            lowestEout = Eout
            C_Eout = C

    print(f"Q9) {C_Ein} gives the lowest value of Ein ({lowestEin}).\n")
    print(f"Q10) {C_Eout} gives the lowest value of Eout ({lowestEout}).\n")


q2()
q3()
q4()
q5_q6()
q7()
q8()
q9_q10()
