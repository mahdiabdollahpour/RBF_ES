import RBF
import data_generator
from utils import save_obj, load_obj
import plots
import matplotlib.pyplot as plt
import numpy as np

x, y = data_generator.load_excel_data("4clstest4000.xlsx")
x_train, y_train = data_generator.load_excel_data("4clstrain1200.xlsx")
y = y - np.ones(shape=(np.shape(y)))
y = y.astype(int)
y_train = y_train - np.ones(shape=(np.shape(y_train)))
y_train = y_train.astype(int)

number_of_circles = 7
dim = 2
from sklearn import preprocessing

ind = load_obj("IND_MULCLS")
W = load_obj("W_MULCLS")
# W = RBF.get_W_multi(x=x_train, y=y_train, individual=ind, m=number_of_circles)
x = preprocessing.scale(x)
x_train = preprocessing.scale(x_train)
# preprocessing.scale(y)

print(
    RBF.multiclass_evaluator(RBF.multiclass_classification_loss, x_train=None, y_train=None, x_test=x, y_test=y, W=W,
                             individual=ind))
y_out = RBF.get_y_multi_classification(individual=ind, x_train=None, y_train=None, x_test=x, W=W)

ax = plt.gca()
ax.cla()  # clear things for fresh plot
for i in range(number_of_circles):
    circle = plt.Circle(ind[i * (dim + 1):i * (dim + 1) + dim], 0.2 / abs(ind[(i + 1) * (dim + 1) - 1]), color='b',
                        fill=False)
    ax.add_artist(circle)

ax.set_xlim((min(2 * min(ind), -2), max(2 * max(ind), 2)))
ax.set_ylim((min(2 * min(ind), -2), max(2 * max(ind), 2)))

# print("ans", ans)
# print(y[1:100])
# print(y_out[1:100])
plots.plot_classif_result(x, y_out=y_out, y=y,mul=True)
