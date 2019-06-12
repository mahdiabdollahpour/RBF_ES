import data_generator
import numpy as np

from utils import save_obj, load_obj

import plots
from sklearn import preprocessing

# x, y = data_generator.moon_data(1000)

x, y = data_generator.load_excel_data("4clstrain1200.xlsx")
y = y - np.ones(shape=(np.shape(y)))
y = y.astype(int)

class_num = 4
print(y[0])
print(min(y))
print(max(y))
print(x[0])
dim = np.shape(x)[1]
number_of_circles = 7

plots.plot_classification_data(x, y, [0, 1, 2, 3])

x = preprocessing.scale(x)
x, y = data_generator.unison_shuffled_copies(x, y)
# print(y[1])
# print(y[2])
# print(y[3])
# print(y[4])
x, x_validation, x_test, y, y_validation, y_test = data_generator.split_data(x, y, 1, 0)

plots.plot_classification_data(x, y, [0, 1, 2, 3])

# x, y = data_generator.classification_data2(number_of_records=500, dim=dim, number_of_classes=class_num)


import OldES
import ES
import RBF

best = ES.find_circle_coordinates(MU=10, LAMBDA=100, evaluator=RBF.multiclass_evaluator,
                                  loss=RBF.multiclass_classification_loss,
                                  NGEN=10, x_train=x,
                                  y_train=y, number_of_circles=number_of_circles, MIN_VALUE=-1.5, MAX_VALUE=1.5,
                                  MIN_STRATEGY=0.5, MAX_STRATEGY=9)

print("best", best)
print("train", RBF.multiclass_evaluator(RBF.multiclass_classification_loss, x, y, x, y, best))

save_obj(np.array(best), "IND_MULCLS")
save_obj(RBF.get_W_multi(x, y, best,number_of_circles), "W_MULCLS")

# print("test", RBF.multiclass_evaluator(RBF.multiclass_classification_loss, x, y, x_test, y_test, best))

ans = RBF.get_y_multi_classification(best, x, y, x)
import matplotlib.pyplot as plt

ax = plt.gca()
ax.cla()  # clear things for fresh plot
for i in range(number_of_circles):
    circle = plt.Circle(best[i * (dim + 1):i * (dim + 1) + dim], 1 / (abs(best[(i + 1) * (dim + 1) - 1])), color='b',
                        fill=False)
    ax.add_artist(circle)

ax.set_xlim((min(2 * min(best), -2), max(2 * max(best), 2)))
ax.set_ylim((min(2 * min(best), -2), max(2 * max(best), 2)))

print("ans", ans)
plots.plot_classification_data(x, ans, [0, 1, 2, 3, 4])
