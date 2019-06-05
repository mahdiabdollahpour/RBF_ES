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


# ~~~~ MODIFICATION TO EXAMPLE BEGINS HERE ~~~~ #
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
import math
from pylab import *
from scipy.interpolate import griddata


def surface_plot(x, y, z):
    # avg = np.mean(z)
    # mu = np.std(z)
    # z = [(x - avg) / (mu * 15) for x in z]

    xyz = {'x': x, 'y': y, 'z': z}

    # put the data into a pandas DataFrame (this is what my data looks like)
    df = pd.DataFrame(xyz, index=range(len(xyz['x'])))

    # re-create the 2D-arrays
    x1 = np.linspace(df['x'].min(), df['x'].max(), len(df['x'].unique()))
    y1 = np.linspace(df['y'].min(), df['y'].max(), len(df['y'].unique()))
    x2, y2 = np.meshgrid(x1, y1)
    z2 = griddata((df['x'], df['y']), df['z'], (x2, y2), method='cubic')

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x2, y2, z2, rstride=1, cstride=1, cmap='RdPu',
                           linewidth=0, antialiased=False)
    ax.set_zlim(-1.01, 1.01)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.title('Meshgrid Created from 3 1D Arrays')
    # ~~~~ MODIFICATION TO EXAMPLE ENDS HERE ~~~~ #

    plt.show()


def scatter_plot(x, y, z):

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter(x, y, z)

    plt.show()
