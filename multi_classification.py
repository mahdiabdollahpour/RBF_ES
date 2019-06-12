import utils
import numpy as np

from utils import save_obj, load_obj

import plots
from sklearn import preprocessing

x, y = utils.load_excel_data("4clstrain1200.xlsx")
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
x, y = utils.unison_shuffled_copies(x, y)

x, x_validation, x_test, y, y_validation, y_test = utils.split_data(x, y, 1, 0)

plots.plot_classification_data(x, y, [0, 1, 2, 3])

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
save_obj(RBF.get_W_multi(x, y, best, number_of_circles), "W_MULCLS")

ans = RBF.get_y_multi_classification(best, x, y, x)
import matplotlib.pyplot as plt

plots.draw_circles(dim, number_of_circles, best)
print("ans", ans)
plots.plot_classification_data(x, ans, [0, 1, 2, 3, 4])
