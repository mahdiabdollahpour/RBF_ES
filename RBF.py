import numpy as np
from params import *

# import math
landa = 1


def g_func(v, sigma, x):
    # print("sigma",sigma)
    return np.exp(-1 * abs(sigma) * np.matmul(np.transpose(x - v), (x - v)))
    ## sigma =1
    # return np.exp(-1 * np.matmul(np.transpose(x - v), (x - v)))


def evaluate(loss, x_data, y_data, individual):
    nums = len(individual)
    dim = len(x_data[0])
    number_of_classes = max(y_data) + 1
    circle_nums = nums / (dim + 1)

    m = int(circle_nums)
    L = len(x_data)
    G = np.zeros(shape=(L, m))
    y_prime = np.zeros(shape=(L, number_of_classes))
    for idx, label in enumerate(y_data):
        y_prime[idx][label] = 1
    for i in range(L):
        for j in range(m):
            # v = np.zeros(shape=(dim))
            v = individual[(dim + 1) * (j - 1):(dim + 1) * (j - 1) + dim]
            G[i][j] = g_func(v, sigma=individual[(dim + 1) * (j - 1) + dim + 1], x=x_data[i])
    W = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(G), G) + landa * np.eye(m)), np.transpose(G)), y_prime)
    y_hat = np.matmul(G, W)
    l = loss(y_prime, y_hat)
    print("how good?", l)
    return -1 * l,


def get_y(individual, x_data, y_data):
    nums = len(individual)
    dim = len(x_data[0])
    number_of_classes = max(y_data) + 1
    circle_nums = nums / (dim + 1)

    m = int(circle_nums)
    L = len(x_data)
    G = np.zeros(shape=(L, m))
    y_prime = np.zeros(shape=(L, number_of_classes))
    for idx, label in enumerate(y_data):
        y_prime[idx][label] = 1
    for i in range(L):
        for j in range(m):
            v = individual[(dim + 1) * (j - 1):(dim + 1) * (j - 1) + dim]
            G[i][j] = g_func(v, sigma=individual[(dim + 1) * (j - 1) + dim + 1], x=x_data[i])
    W = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(G), G) + landa * np.eye(m)), np.transpose(G)), y_prime)
    y_hat = np.matmul(G, W)
    # print("y_hat", y_hat)
    y = np.argmax(y_hat, axis=1)
    return y


def regression_loss(y, y_hat):
    return (1 / 2) * np.dot(np.transpose(y - y_hat), (y_hat - y))


def classification_loss(y, y_hat):
    # print("y_hat", np.shape(y_hat), "y", np.shape(y), len(y_hat))

    return 1 - ((np.sum(np.sign(np.abs(np.argmax(y, axis=1) - np.argmax(y_hat, axis=1))))) / len(y_hat))
