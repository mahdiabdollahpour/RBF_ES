import RBF
import utils
from utils import save_obj, load_obj
import plots
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

x_train, y_train = utils.load_excel_data("2clstrain1200.xlsx")
x, y = utils.load_excel_data("2clstest4000.xlsx")

plots.plot_classification_data(x, y, [-1, 1])

number_of_circles = 7
dim = len(x[0])
print("dim", len(x[0]))

ind = load_obj("IND_2CLS")
W = RBF.get_W(x_train, y_train, ind, number_of_circles)
W = load_obj("W_2CLS")
print(ind)
x = preprocessing.scale(x)
x_train = preprocessing.scale(x_train)
# preprocessing.scale(y)

print(
    RBF.evaluator(RBF.binary_classification_loss, x_train=x_train, y_train=y_train, x_test=x, y_test=y, W=W,
                  individual=ind))

y_out = RBF.get_y_binary_classification(individual=ind, x_train=None, y_train=None, x_test=x, W=W)
print(y_out)
ax = plt.gca()
ax.cla()  # clear things for fresh plot
for i in range(number_of_circles):
    circle = plt.Circle(ind[i * (dim + 1):i * (dim + 1) + dim], 1 / abs(ind[(i + 1) * (dim + 1) - 1]), color='b',
                        fill=False)
    ax.add_artist(circle)

ax.set_xlim((min(2 * min(ind), -2), max(2 * max(ind), 2)))
ax.set_ylim((min(2 * min(ind), -2), max(2 * max(ind), 2)))

# print("ans", ans)
# plots.plot_classification_data(x, y_out, [1, 0])
plots.plot_classif_result(x=x, y_out=y_out, y=y)
