import data_generator
import numpy as np

import plots
from sklearn import preprocessing

# x, y = data_generator.moon_data(1000)
x, y = data_generator.load_excel_data("2clstrain.xlsx")

class_num = 2
dim = np.shape(x)[1]
number_of_circles = 20

plots.plot_classification_data(x, y, [-1, 1])

x = preprocessing.scale(x)

plots.plot_classification_data(x, y, [-1, 1])

# x, y = data_generator.classification_data2(number_of_records=500, dim=dim, number_of_classes=class_num)

x, y = data_generator.unison_shuffled_copies(x, y)
x, x_validation, x_test, y, y_validation, y_test = data_generator.split_data(x, y, 0.6, 0)

import OldES
import NewEs
import RBF

best = NewEs.find_circle_coordinates(evaluator=RBF.evaluator, loss=RBF.binary_classification_loss, NGEN=20, x_train=x,
                                     y_train=y, number_of_circles=number_of_circles)

print("best", best)
print("train", RBF.evaluator(RBF.binary_classification_loss, x, y, x, y, best))
print("test", RBF.evaluator(RBF.binary_classification_loss, x, y, x_test, y_test, best))

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
plots.plot_classification_data(x_test, ans, [-1, 1])
