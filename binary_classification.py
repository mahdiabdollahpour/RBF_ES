import utils
import numpy as np

import plots
from sklearn import preprocessing
from utils import save_obj, load_obj

# x, y = utils.moon_data(1000)

x_train, y_train = utils.load_excel_data("2clstrain1200.xlsx")

class_num = 2
dim = np.shape(x_train)[1]
number_of_circles = 7

x_train = preprocessing.scale(x_train)
plots.plot_classification_data(x_train, y_train, [-1, 1])

# x, y = utils.unison_shuffled_copies(x, y)
# x, x_validation, x_test, y, y_validation, y_test = utils.split_data(x, y, 1, 0)

plots.plot_classification_data(x_train, y_train, [-1, 1])

import ES
import RBF

ind = ES.find_circle_coordinates(MU=10, LAMBDA=100, evaluator=RBF.evaluator, loss=RBF.binary_classification_loss,
                                 NGEN=8, x_train=x_train,
                                 y_train=y_train, number_of_circles=number_of_circles, MIN_VALUE=-1.5, MAX_VALUE=1.5,
                                 MIN_STRATEGY=0.5, MAX_STRATEGY=9)

print("best", ind)
save_obj(list(ind), "IND_2CLS")
# print(list(best))
save_obj(RBF.get_W(x_train, y_train, ind, number_of_circles), "W_2CLS")
# print("train", RBF.evaluator(RBF.binary_classification_loss, x_train, y_train, x, y, ind))
# print("test", RBF.evaluator(RBF.binary_classification_loss, x, y, x_test, y_test, best))

ans = RBF.get_y_binary_classification(ind, x_train, y_train, x_train)
import matplotlib.pyplot as plt

plots.draw_circles(dim, number_of_circles, ind)
print("ans", ans)

plots.plot_classification_data(x_train, ans, [1, 0])
