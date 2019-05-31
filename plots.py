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


def plot_classification_data(x, y):
    class1 = np.array([x[i] for i in range(len(x)) if y[i] == 0])
    class2 = np.array([x[i] for i in range(len(x)) if y[i] == 1])
    class3 = np.array([x[i] for i in range(len(x)) if y[i] == 2])

    plot_points([class1, class2, class3], ['ro', 'bo', 'go'])
