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


def SVM_run(kernel_type, classifiers, C=0.01, degree=2):
    errors_Ein = []
    errors_Eout = []
    model = None
    for c in classifiers:
        # 1 versus all classifier:
        temp_y_train = np.array(
            [1 if c == classifier else -1 for classifier in y_train])
        temp_y_test = np.array(
            [1 if c == classifier else -1 for classifier in y_train])

        #  (1 + x_n^Tx_m)^Q, here 1 is coef0, so we set coef0 as 1:
        model = svm.SVC(C=C, degree=degree, kernel=kernel_type,
                        coef0=1, gamma=1)
        model.fit(x_train, temp_y_train)
        result = model.score(x_train, temp_y_train)
        Ein = (1 - result) * 100  # Ein = 1 - accuracy
        errors_Ein.append(Ein)

        result = model.score(x_test, temp_y_test)
        Eout = (1 - result) * 100  # Eout = 1 - accuracy
        errors_Eout.append(Ein)

    return errors_Eout, errors_Ein, model


def q2():
    classifiers = [0, 2, 4, 6, 8]
    _, errors, _ = SVM_run("poly", classifiers)
    res = np.argmax(errors)

    print(
        f"Q2) The classifier {classifiers[res]} has the highest Ein among others with Ein: {max(errors)}%")


def q3():
    classifiers = [1, 3, 5, 7, 9]
    _, errors, _ = SVM_run("poly", classifiers)
    res = np.argmin(errors)
    print(
        f"Q3) The classifier {classifiers[res]} has the lowest Ein among others with Ein: {min(errors)}%")


def q4():
    classifiers = [0]
    _, _, model_0 = SVM_run("poly", classifiers)

    classifiers = [1]
    _, _, model_1 = SVM_run("poly", classifiers)

    # @len(model_0.support_vectors_) returns the number of support vectors for each class
    res = len(model_0.support_vectors_) - len(model_1.support_vectors_)
    print(
        f"Q4) Difference between the number of support vectors of these two classifiers is {res}")


# For Q5, answer d must be correct because the maximum C
# achieves the lowest Ein and all other values do not regularly
# decrease as it is mentioned in the other answers.


def q5_q6():
    print("Q5)")
    Cs = [0.001, 0.01, 0.1, 1]
    classifiers = [1, 5]

    temp_y_train = []
    for c in y_train:
        classifier = 0
        if c == classifiers[0]:
            classifier = 1
        elif c == classifiers[1]:
            classifier = -1
        temp_y_train.append(classifier)

    temp_y_test = []
    for c in y_test:
        classifier = 0
        if c == classifiers[0]:
            classifier = 1
        elif c == classifiers[1]:
            classifier = -1
        temp_y_test.append(classifier)

    temp_y_train = np.array(temp_y_train)
    temp_y_test = np.array(temp_y_test)

    for C in Cs:
        model = svm.SVC(C=C, degree=2, kernel="poly",
                        coef0=1, gamma=1)
        model.fit(x_train, temp_y_train)
        result = model.score(x_train, temp_y_train)
        Ein = (1 - result) * 100

        result = model.score(x_test, temp_y_test)
        Eout = (1 - result) * 100
        print(
            f"For C={C}: Ein={Ein}%, Eout={Eout}, Number of support vectors:{len(model.support_vectors_)} ")

    print("Q6)")
    Cs = [0.0001, 0.001, 0.01, 0.1, 1]
    Qs = [2, 5]
    for Q in Qs:
        for C in Cs:
            model = svm.SVC(C=C, degree=Q, kernel="poly",
                            coef0=1, gamma=1)
            model.fit(x_train, temp_y_train)
            result = model.score(x_train, temp_y_train)
            Ein = (1 - result) * 100

            result = model.score(x_test, temp_y_test)
            Eout = (1 - result) * 100
            print(
                f"For C={C} and Q={Q}: Ein={Ein}%, Eout={Eout}, Number of support vectors:{len(model.support_vectors_)} ")


def q7():
    print("Q7)")
    Cs = [0.0001, 0.001, 0.01, 0.1, 1]
    classifiers = [1, 5]
    size = len(temp_x_train)
    temp_y_train = []
    temp_x_train = x_train.copy()
    for c in y_train:
        classifier = 0
        if c == classifiers[0]:
            classifier = 1
        elif c == classifiers[1]:
            classifier = -1
        temp_y_train.append(classifier)

    selectedOnes = {0.0001: 0, 0.001: 0, 0.01: 0, 0.1: 0, 1: 0}

    for i in range(100):
        lowestEin = 100
        bestC = 0
        for C in Cs:
            rand = random.randint(int(size / 10), int(9 * size / 10))
            test_X = temp_x_train[rand:rand + int(size / 10)]
            test_Y = temp_y_train[rand:rand + int(size / 10)]
            model = svm.SVC(C=C, degree=2, kernel="poly",
                            coef0=1, gamma=1)
            model.fit(temp_x_train, temp_y_train)
            result = model.score(test_X, test_Y)
            Ein = (1 - result) * 100
            if Ein < lowestEin:
                lowestEin = Ein
                bestC = C
        selectedOnes[bestC] += 1

    print("The selection numbers of C's:")
    print(selectedOnes)


def q8():
    print("Q8)")
    C = 0.001
    classifiers = [1, 5]
    size = len(temp_x_train)
    temp_y_train = []
    temp_x_train = x_train.copy()
    size = len(temp_x_train)
    for c in y_train:
        classifier = 0
        if c == classifiers[0]:
            classifier = 1
        elif c == classifiers[1]:
            classifier = -1
        temp_y_train.append(classifier)

    meanEcv = 0
    for i in range(100):
        rand = random.randint(int(size / 10), int(9 * size / 10))
        test_X = temp_x_train[rand:rand + int(size / 10)]
        test_Y = temp_y_train[rand:rand + int(size / 10)]
        model = svm.SVC(C=C, degree=2, kernel="poly",
                        coef0=1, gamma=1)
        model.fit(temp_x_train, temp_y_train)
        result = model.score(test_X, test_Y)
        Ein = (1 - result) * 100
        meanEcv += Ein
    meanEcv /= 100

    print(f"The average value of Ecv over the 100 runs is{meanEcv}")


def q9():
    print("Q9)")
    Cs = [0.01, 1, 100, 10**4, 10**6]
    classifiers = [1, 5]
    temp_y_train = []
    temp_x_train = x_train.copy()
    for c in y_train:
        classifier = 0
        if c == classifiers[0]:
            classifier = 1
        elif c == classifiers[1]:
            classifier = -1
        temp_y_train.append(classifier)

    lowestEin = 100
    bestC = 0
    for C in Cs:
        model = svm.SVC(C=C, degree=2, kernel="rbf",
                        coef0=1, gamma=1)
        model.fit(temp_x_train, temp_y_train)
        result = model.score(temp_x_train, temp_y_train)
        Ein = (1 - result) * 100
        if Ein < lowestEin:
            lowestEin = Ein
            bestC = C

    print(f"{bestC} gives the lowest value of Ein ({lowestEin}).")


def q10():
    print("Q10)")
    Cs = [0.01, 1, 100, 10**4]
    classifiers = [1, 5]
    temp_y_train = []
    temp_x_train = x_train.copy()
    for c in y_train:
        classifier = 0
        if c == classifiers[0]:
            classifier = 1
        elif c == classifiers[1]:
            classifier = -1
        temp_y_train.append(classifier)

    temp_y_test = []
    temp_x_test = x_test.copy()
    for c in y_test:
        classifier = 0
        if c == classifiers[0]:
            classifier = 1
        elif c == classifiers[1]:
            classifier = -1
        temp_y_test.append(classifier)

    lowestEout = 100
    bestC = 0
    for C in Cs:
        model = svm.SVC(C=C, degree=2, kernel="rbf",
                        coef0=1, gamma=1)
        model.fit(temp_x_train, temp_y_train)
        result = model.score(temp_x_test, temp_y_test)
        Eout = (1 - result) * 100
        if Eout < lowestEout:
            lowestEout = Eout
            bestC = C

    print(f"{bestC} gives the lowest value of Eout ({lowestEout}).")


q9()
q10()
