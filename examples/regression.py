import data_generator
import numpy as np


# x, y = data_generator.regression_data2(1000, dim)
x, y = data_generator.load_excel_data("regdata2000.xlsx")

dim = np.shape(x)[1]
number_of_circles = 10

import plots
from sklearn import preprocessing

# min_max_scaler = preprocessing.MinMaxScaler()
# x = min_max_scaler.fit_transform(x)

x = preprocessing.scale(x)
y = preprocessing.scale(y)

print(np.shape(x))
print(np.shape(y))

x, y = data_generator.unison_shuffled_copies(x, y)
x, x_validation, x_test, y, y_validation, y_test = data_generator.split_data(x, y, 0.6, 0)
# plots.scatter_plot(x[:, 0], x[:, 1], y)

import OldES
import RBF

best = OldES.find_circle_coordinates(x, y, number_of_circles, RBF.evaluator, RBF.regression_loss, NGEN=20, POPSIZE=30)
print("best", best)

print(RBF.evaluator(RBF.regression_loss, x, y, x_test, y_test, best))

ans = RBF.get_y_regression(best, x, y, x_test)

plots.scatter_plot(x_test[:, 0], x_test[:, 1], ans)
