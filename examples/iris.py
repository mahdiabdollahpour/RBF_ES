from sklearn import datasets

import data_generator
import numpy as np

x, y = datasets.load_wine(True)
print(x[0])
print(y[0])
class_num = max(y) + 1
dim = np.shape(x)[1]
print("Classes", class_num)
print("Dimention", dim)

number_of_circles = 4

from sklearn import preprocessing
import plots

# x, y = data_generator.classification_data(500, dim, class_num)
# plots.plot_classification_data(x, y)

x = preprocessing.scale(x)

# plots.plot_classification_data(x, y)

x, y = data_generator.unison_shuffled_copies(x, y)

x, x_validation, x_test, y, y_validation, y_test = data_generator.split_data(x, y, 0.8, 0)

import OldES
import RBF

best = OldES.find_circle_coordinates(x, y, number_of_circles, RBF.multiclass_evaluator, RBF.multiclass_classification_loss,
                                     NGEN=50, POPSIZE=10)

print("best", best)

print("test :", RBF.multiclass_evaluator(RBF.multiclass_classification_loss, x, y, x_test, y_test, best))
print("train :", RBF.multiclass_evaluator(RBF.multiclass_classification_loss, x, y, x, y, best))

ans = RBF.get_y_multi_classification(best, x, y, x_test)

print("ans", ans)

#
# import matplotlib.pyplot as plt
#
# ax = plt.gca()
# ax.cla()  # clear things for fresh plot
# for i in range(number_of_circles):
#     circle = plt.Circle(best[i * (dim + 1):i * (dim + 1) + dim], abs(best[(i + 1) * (dim + 1) - 1]), color='b',
#                         fill=False)
#     ax.add_artist(circle)
#
# ax.set_xlim((min(2 * min(best), -2), max(2 * max(best), 2)))
# ax.set_ylim((min(2 * min(best), -2), max(2 * max(best), 2)))
#
# plots.plot_classification_data(x_test, ans)
