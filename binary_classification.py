import data_generator
import numpy as np

class_num = 2
dim = 2
number_of_circles = 4
import plots
from sklearn import preprocessing

x, y = data_generator.moon_data(1000)
plots.plot_classification_data(x, y)

x = preprocessing.scale(x)

plots.plot_classification_data(x, y)

# x, y = data_generator.classification_data2(number_of_records=500, dim=dim, number_of_classes=class_num)

x, y = data_generator.unison_shuffled_copies(x, y)
x, x_validation, x_test, y, y_validation, y_test = data_generator.split_data(x, y)

import ES
import RBF

best = ES.find_circle_coordinates(x, y, number_of_circles, RBF.evaluator, RBF.binary_classification_loss)

print("best", best)
print(RBF.evaluator(RBF.binary_classification_loss, x, y, x_test, y_test, best))

ans = RBF.get_y_binary_classification(best, x, y, x_test)
import matplotlib.pyplot as plt

ax = plt.gca()
ax.cla()  # clear things for fresh plot
for i in range(number_of_circles):
    circle = plt.Circle(best[i * (dim + 1):i * (dim + 1) + dim], abs(best[(i + 1) * (dim + 1) - 1]), color='b',
                        fill=False)
    ax.add_artist(circle)

ax.set_xlim((min(2 * min(best), -2), max(2 * max(best), 2)))
ax.set_ylim((min(2 * min(best), -2), max(2 * max(best), 2)))

print("ans", ans)
plots.plot_classification_data(x_test, ans)
