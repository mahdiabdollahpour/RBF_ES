import pickle


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)



import numpy as np
from sklearn.datasets import make_moons
from sklearn.datasets import make_regression
import xlrd


def load_excel_data(file_name):
    loc = (file_name)
    wb = xlrd.open_workbook(loc)
    sheet = wb.sheet_by_index(0)
    sheet.cell_value(0, 0)
    data = np.zeros((sheet.nrows, sheet.ncols))
    for i in range(sheet.nrows):
        for j in range(sheet.ncols):
            data[i, j] = sheet.cell_value(i, j)
    data = np.array(data)
    y = data[:, data.shape[1] - 1]

    x = data[:, :data.shape[1] - 1]
    return x, y


import pandas as pd


def load_CSV_data(file_name):
    # loc = (file_name)
    data = pd.read_csv(file_name)

    data = np.array(data)
    # print(data)
    y = data[:, data.shape[1] - 1]

    x = data[:, :data.shape[1] - 1]
    return x, y





def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def split_data(x, y, a, b):
    r = len(x)
    train = int(a * r)
    validation = int(b * r)
    # test = r - validation - train
    x_train = x[:train]
    x_validation = x[train:train + validation]
    x_test = x[train + validation:]
    y_train = y[:train]
    y_validation = y[train:train + validation]
    y_test = y[train + validation:]
    return x_train, x_validation, x_test, y_train, y_validation, y_test
