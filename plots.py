import matplotlib.pyplot as plt
import numpy as np


def plot_points(x, colors):
    for i, data in enumerate(x):
        # print(data)
        if len(data) > 0:
            g_d0 = data[:, 0]
            g_d1 = data[:, 1]
            plt.plot(g_d0, g_d1, colors[i])
    plt.show()


def plot_regression_result(y_correct, y_model):
    plt.plot(range(len(y_correct)), y_correct, 'bo')
    plt.plot(range(len(y_model)), y_model, 'ro')
    plt.show()


def plot_classif_result(x, y_out, y):
    right = np.array([x[i] for i in range(len(x)) if y_out[i] == y[i]])
    wrong = np.array([x[i] for i in range(len(x)) if y_out[i] != y[i]])
    plot_points([wrong, right], ['ro', 'go'])


def draw_circles(dim, number_of_circles, ind):
    ax = plt.gca()
    ax.cla()  # clear things for fresh plot
    for i in range(number_of_circles):
        circle = plt.Circle(ind[i * (dim + 1):i * (dim + 1) + dim], abs(ind[(i + 1) * (dim + 1) - 1]), color='b',
                            fill=False)
        ax.add_artist(circle)

    ax.set_xlim((min(3 * min(ind), -3), max(3 * max(ind), 3)))
    ax.set_ylim((min(3 * min(ind), -3), max(3 * max(ind), 3)))


def plot_classification_data(x, y, classes):
    list = []
    for j in range(len(classes)):
        classj = np.array([x[i] for i in range(len(x)) if y[i] == classes[j]])
        list.append(classj)

    plot_points(list, ['ro', 'bo', 'go', 'yo', 'ko'])
