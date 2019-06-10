import numpy as np
from params import *

# import math
landa = 1


# def inv_covariance_matrix(x, v, N):
#     return (1 / N) * np.linalg.inv(np.matmul(np.transpose(x - v), (x - v)))


def g_func(v, sigma, x, N):
    # print("sigma",sigma)
    # inv_cov = inv_covariance_matrix(x, v, N)
    return np.exp(-1 * sigma * np.matmul(np.transpose(x - v), (x - v)))


def get_G_matrix(x, individual):
    nums = len(individual)
    dim = len(x[0])
    circle_nums = nums / (dim + 1)
    m = int(circle_nums)
    L = len(x)
    G = np.zeros(shape=(L, m))
    for i in range(L):
        for j in range(m):
            v = individual[(dim + 1) * (j - 1):(dim + 1) * (j - 1) + dim]
            G[i][j] = g_func(v, sigma=individual[(dim + 1) * (j - 1) + dim + 1], x=x[i], N=L)
    return G


def multiclass_evaluator(loss, x_train, y_train, x_test, y_test, individual):
    nums = len(individual)
    dim = len(x_train[0])
    number_of_classes = max(y_train) + 1
    circle_nums = nums / (dim + 1)

    m = int(circle_nums)
    L = len(x_train)
    # G = np.zeros(shape=(L, m))
    y_prime_train = np.zeros(shape=(L, number_of_classes))
    y_prime_test = np.zeros(shape=(len(x_test), number_of_classes))
    for idx, label in enumerate(y_train):
        y_prime_train[idx][label] = 1
    for idx, label in enumerate(y_test):
        y_prime_test[idx][label] = 1
    G = get_G_matrix(x_train, individual)
    G2 = get_G_matrix(x_test, individual)

    W = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(G), G) + landa * np.eye(m)), np.transpose(G)),
                  y_prime_train)
    y_hat = np.matmul(G2, W)
    # print(np.shape(y_hat))
    # print(np.shape(y_prime_test))
    l = loss(y_prime_test, y_hat)
    print("Accuracy:", l)
    return -1 * l,


def evaluator(loss, x_train, y_train, x_test, y_test, individual):
    nums = len(individual)
    dim = len(x_train[0])
    circle_nums = nums / (dim + 1)
    m = int(circle_nums)

    G = get_G_matrix(x_train, individual)
    G2 = get_G_matrix(x_test, individual)

    W = np.matmul(np.matmul(np.linalg.pinv(np.matmul(np.transpose(G), G) + landa * np.eye(m)), np.transpose(G)), y_train)
    y_hat = np.matmul(G2, W)
    l = loss(y_test, y_hat)
    print("Fitness:", l)
    return -l,


def get_y_multi_classification(individual, x_train, y_train, x_test):
    nums = len(individual)
    dim = len(x_train[0])
    circle_nums = nums / (dim + 1)
    m = int(circle_nums)
    number_of_classes = max(y_train) + 1
    y_prime_train = np.zeros(shape=(len(x_train), number_of_classes))
    for idx, label in enumerate(y_train):
        y_prime_train[idx][label] = 1
    G = get_G_matrix(x_train, individual)
    G2 = get_G_matrix(x_test, individual)
    W = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(G), G) + landa * np.eye(m)), np.transpose(G)),
                  y_prime_train)
    y_hat = np.matmul(G2, W)
    y = np.argmax(y_hat, axis=1)
    return y


def get_y_regression(individual, x_train, y_train, x_test):
    nums = len(individual)
    dim = len(x_train[0])
    circle_nums = nums / (dim + 1)
    m = int(circle_nums)
    G = get_G_matrix(x_train, individual)
    G2 = get_G_matrix(x_test, individual)
    W = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(G), G) + landa * np.eye(m)), np.transpose(G)), y_train)
    y_hat = np.matmul(G2, W)
    return y_hat


def get_y_binary_classification(individual, x_train, y_train, x_test):
    nums = len(individual)
    dim = len(x_train[0])
    # number_of_classes = max(y_data) + 1
    circle_nums = nums / (dim + 1)

    m = int(circle_nums)
    # L = len(x_train)
    G = get_G_matrix(x_train, individual)
    G2 = get_G_matrix(x_test, individual)

    W = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(G), G) + landa * np.eye(m)), np.transpose(G)), y_train)
    # predicted for x_test
    y_hat = np.heaviside(np.matmul(G2, W), 0)

    return y_hat


def regression_loss(y, y_hat):
    return (-1 / (2 * len(y))) * np.dot(np.transpose(y_hat - y), (y_hat - y))


def multiclass_classification_loss(y, y_hat):
    return 1 - ((np.sum(np.sign(np.abs(np.argmax(y, axis=1) - np.argmax(y_hat, axis=1))))) / len(y_hat))


def binary_classification_loss(y_hat, y):
    return 1 - ((np.sum(np.abs(np.sign(y) - y_hat))) / len(y_hat))
