import math
import sys

from copy import deepcopy
import numpy as np
import pandas as pd


def sigmoid_func(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid_func(x) * (1 - sigmoid_func(x))


def loss_func(y, yhat):
    loss = y.values * np.log(yhat + 1e-8) + (1 - y.values) * np.log(1 - yhat + 1e-8)
    return (loss.sum(axis=1).sum(axis=0)) / -y.shape[0]


def fp_func(train_x, weights):
    outputs = {}
    activations = {}
    temp = deepcopy(train_x)
    activations["A0"] = train_x
    num = int(len(weights) / 2) + 1
    for i in range(1, num):
        x1 = temp.dot(weights["W" + str(i)].values)
        x = pd.DataFrame(x1.values + weights["b" + str(i)].values)
        if i == len(weights) / 2:
            temp = (np.exp(x.transpose()) / np.exp(x.transpose()).sum()).transpose()
        else:
            temp = x.apply(sigmoid_func)
        outputs["Out" + str(i)] = x
        activations["A" + str(i)] = temp
    return outputs, activations


def bk_func(weights, acti, opt, y, lr):
    delta = {}
    acti_oneless = len(acti) - 1
    Ah = acti["A" + str(len(acti) - 2)]
    Aout = acti["A" + str(acti_oneless)]
    delta["d" + str(acti_oneless)] = pd.DataFrame(Aout.values - y.values)
    change = lr * Ah.transpose().dot(delta["d" + str(acti_oneless)])
    weights["W" + str(acti_oneless)] -= change
    weights["b" + str(acti_oneless)] -= lr * delta["d" + str(acti_oneless)].sum(axis=0)
    for i in range(len(acti) - 2, 0, -1):
        delta_plus1 = delta["d" + str(i + 1)]
        w_plus1 = weights["W" + str(i + 1)]
        gx = sigmoid_derivative(opt["Out" + str(i)])
        delta["d" + str(i)] = pd.DataFrame(
            gx.values * (delta_plus1.values.dot(w_plus1.transpose().values))
        )
        change = lr * acti["A" + str(i - 1)].transpose().values.dot(delta["d" + str(i)])
        weights["W" + str(i)] -= change
        weights["b" + str(i)] -= lr * delta["d" + str(i)].sum(axis=0)
    delta.clear()
    return weights


def train_nn(train_x, train_y, size, learning_rate, epoch, batch_size):
    # Weights setup
    weights = {}
    m1 = size.pop(0)
    sigma = math.sqrt(3 / m1)
    for i in range(len(size)):
        weights["W" + str(i + 1)] = pd.DataFrame(np.random.randn(m1, size[i]) * sigma)
        weights["b" + str(i + 1)] = pd.DataFrame(np.zeros(shape=(1, size[i])))
        m1 = size[i]
        sigma = math.sqrt(3 / m1)
    y = pd.DataFrame(np.zeros((train_y.shape[0], 10)))
    for i in range(len(train_y)):
        y[train_y[0][i]][i] = 1
    errors = []
    for i in range(0, epoch):
        batch_dash = 0
        while batch_dash < train_x.shape[0]:
            Outputs, Activations = fp_func(
                train_x[batch_dash : batch_dash + batch_size], weights
            )
            loss = loss_func(
                y[batch_dash : batch_dash + batch_size],
                Activations["A" + str(len(Activations) - 1)],
            )
            errors.append(loss)
            weights = bk_func(
                weights,
                Activations,
                Outputs,
                y[batch_dash : batch_dash + batch_size],
                learning_rate,
            )
            batch_dash += batch_size
    return weights


def predict(x, weights):
    output, Activations = fp_func(x, weights)
    y = Activations["A" + str(len(Activations) - 1)]
    yh = pd.DataFrame(y.idxmax(axis=1))
    return yh


def accuracy(y, yhat):
    correct = y.shape[0] - np.count_nonzero((y - yhat).values)
    return correct / y.shape[0]


if __name__ == "__main__":
    train_x_file = sys.argv[1]
    train_y_file = sys.argv[2]
    test_x_file = sys.argv[3]
    train_x = pd.read_csv(train_x_file, header=None)
    train_y = pd.read_csv(train_y_file, header=None)
    test_x = pd.read_csv(test_x_file, header=None)

    learning_rate = 0.0004
    epoch = 100
    batch_size = 500
    size = [train_x.shape[1], 250, 80, 10]
    weights = train_nn(
        train_x, train_y, size, learning_rate, epoch, batch_size
    )
    y = predict(train_x, weights)
    acc = accuracy(train_y, y, verbose=True)
    print("Training Accuracy = ", acc * 100, "%")
    y = predict(test_x, weights)
    y.to_csv("test_predictions.csv", index=False, header=False)

