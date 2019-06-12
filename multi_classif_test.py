import RBF
import utils
from utils import save_obj, load_obj
import plots
import matplotlib.pyplot as plt
import numpy as np

x, y = utils.load_excel_data("4clstest4000.xlsx")
x_train, y_train = utils.load_excel_data("4clstrain1200.xlsx")
y = y - np.ones(shape=(np.shape(y)))
y = y.astype(int)
y_train = y_train - np.ones(shape=(np.shape(y_train)))
y_train = y_train.astype(int)

number_of_circles = 7
dim = 2
from sklearn import preprocessing

ind = load_obj("IND_MULCLS")
W = load_obj("W_MULCLS")
x = preprocessing.scale(x)
x_train = preprocessing.scale(x_train)

print(
    RBF.multiclass_evaluator(RBF.multiclass_classification_loss, x_train=None, y_train=None, x_test=x, y_test=y, W=W,
                             individual=ind))
y_out = RBF.get_y_multi_classification(individual=ind, x_train=None, y_train=None, x_test=x, W=W)

plots.draw_circles(dim, number_of_circles, ind)

plots.plot_classif_result(x, y_out=y_out, y=y)
