import data_generator
import numpy as np

class_num = 3
dim = 2

x, y = data_generator.classification_data(number_of_records=500, dim=dim, number_of_classes=class_num)
# print(np.argwhere(y))
# print(y)

import plots
from sklearn import preprocessing

plots.plot_classification_data(x, y)

# min_max_scaler = preprocessing.MinMaxScaler()
# x = min_max_scaler.fit_transform(x)
x = preprocessing.scale(x)
# x.mean(axis=0)
# x.std(axis=0)
print(x)
plots.plot_classification_data(x, y)

import ES
import RBF

number_of_circles = class_num
chromosome_size = (dim + 1) * number_of_circles

es = ES.ES(50, RBF.evaluate, RBF.classification_loss, x, y, chromosome_size)

pop = es.solve_problem(NGEN=1)

best = es.get_best_in_pop(pop)
print("best", best)
print(RBF.evaluate(RBF.classification_loss, x, y, best))

ans = RBF.get_y(best, x, y)
import matplotlib.pyplot as plt

ax = plt.gca()
ax.cla()  # clear things for fresh plot
for i in range(number_of_circles):
    circle = plt.Circle(best[i * (dim + 1):i * (dim + 1) + dim], abs(best[(i + 1) * (dim + 1) - 1]), color='b',
                        fill=False)
    ax.add_artist(circle)

ax.set_xlim((2 * min(best), 2 * max(best)))
ax.set_ylim((2 * min(best), 2 * max(best)))

print("ans", ans)
plots.plot_classification_data(x, ans)
