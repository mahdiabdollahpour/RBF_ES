import data_generator
import numpy as np


class_num = 2
dim = 2
number_of_circles = 10

# x, y = data_generator.classification_data2(number_of_records=500, dim=dim, number_of_classes=class_num)
x, y = data_generator.regression_data2(50, dim)

x, y = data_generator.unison_shuffled_copies(x, y)
x, x_validation, x_test, y, y_validation, y_test = data_generator.split_data(x, y)

import plots
from sklearn import preprocessing

# min_max_scaler = preprocessing.MinMaxScaler()
# x = min_max_scaler.fit_transform(x)
x = preprocessing.scale(x)
y = preprocessing.scale(y)
plots.scatter_plot(x[:, 0], x[:, 1], y)

import ES
import RBF

best = ES.find_circle_coordinates(x, y, number_of_circles, RBF.evaluator, RBF.regression_loss)
print("best", best)

print(RBF.evaluator(RBF.regression_loss, x, y, x_test, y_test, best))

plots.scatter_plot(x[:, 0], x[:, 1], y)


ans = RBF.get_y_regression(best, x, y, x_test)

