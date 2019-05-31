import data_generator
import numpy as np

## TODO: clear the code, data and test or regression, split data, change sigma for ES from max to min by NGEN
class_num = 2
dim = 2
number_of_circles = 10

# x, y = data_generator.classification_data2(number_of_records=500, dim=dim, number_of_classes=class_num)
x, y = data_generator.regression_data2(500, dim)

x, y = data_generator.unison_shuffled_copies(x, y)
x, x_validation, x_test, y, y_validation, y_test = data_generator.split_data(x, y)

import plots
from sklearn import preprocessing

# plots.plot_classification_data(x, y)

# min_max_scaler = preprocessing.MinMaxScaler()
# x = min_max_scaler.fit_transform(x)
x = preprocessing.scale(x)
y = preprocessing.scale(y)
# x.mean(axis=0)
# x.std(axis=0)
# print(x)
# plots.plot_classification_data(x, y)

import ES
import RBF

chromosome_size = (dim + 1) * number_of_circles

es = ES.ES(50, RBF.regression_evaluator, RBF.regression_loss, x, y, chromosome_size, min_sigma_mut=0.4, max_sigma_mut=0.9,
           indpb_mut=0.1)

pop = es.solve_problem(NGEN=10)

best = es.get_best_in_pop(pop)
print("best", best)
print(RBF.regression_evaluator(RBF.regression_loss, x, y, best))

ans = RBF.get_y_regression(best, x, y)
import matplotlib.pyplot as plt

# ax = plt.gca()
# ax.cla()  # clear things for fresh plot
# for i in range(number_of_circles):
#     circle = plt.Circle(best[i * (dim + 1):i * (dim + 1) + dim], abs(best[(i + 1) * (dim + 1) - 1]), color='b',
#                         fill=False)
#     ax.add_artist(circle)
#
# ax.set_xlim((min(2 * min(best), -2), max(2 * max(best), 2)))
# ax.set_ylim((min(2 * min(best), -2), max(2 * max(best), 2)))

# print("ans", ans)
# plots.plot_classification_data(x, ans)
