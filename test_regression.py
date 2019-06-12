import RBF
import data_generator
from utils import save_obj, load_obj
import plots

x, y = data_generator.load_excel_data("regdata1500.xlsx")

from sklearn import preprocessing

ind = load_obj("IND_REG")
W = load_obj("W_REG")

x = preprocessing.scale(x)
y = preprocessing.scale(y)

print(RBF.evaluator(RBF.regression_loss, x_train=None, y_train=None, x_test=x, y_test=y, W=W, individual=ind))
y_out = RBF.get_y_regression(individual=ind, x_train=None, y_train=None, x_test=x, W=W)

plots.plot_regression_result(y_correct=y, y_model=y_out)
