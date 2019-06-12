import utils
import numpy as np
from utils import save_obj, load_obj

# x, y = utils.regression_data2(1000, dim)
x_train, y_train = utils.load_excel_data("regdata1500.xlsx")

dim = np.shape(x_train)[1]
number_of_circles = 20

import plots
from sklearn import preprocessing

# min_max_scaler = preprocessing.MinMaxScaler()
# x = min_max_scaler.fit_transform(x)

x_train = preprocessing.scale(x_train)
y_train = preprocessing.scale(y_train)

print(np.shape(x_train))
print(np.shape(y_train))

x_train, y_train = utils.unison_shuffled_copies(x_train, y_train)
x_train, x_validation, x_test, y_train, y_validation, y_test = utils.split_data(x_train, y_train, 0.6, 0)
# plots.scatter_plot(x[:, 0], x[:, 1], y)

import ES
import RBF

best = ES.find_circle_coordinates(MU=10, LAMBDA=100, evaluator=RBF.evaluator, loss=RBF.regression_loss,
                                  NGEN=10, x_train=x_train,
                                  y_train=y_train, number_of_circles=number_of_circles, MIN_VALUE=-1.5, MAX_VALUE=1.5,
                                  MIN_STRATEGY=0.5, MAX_STRATEGY=9)

print("best", best)
save_obj(np.array(best), "IND_REG")
save_obj(RBF.get_W(x_train, y_train, best, number_of_circles), "W_REG")
print(RBF.evaluator(RBF.regression_loss, x_train, y_train, x_train, y_train, best))

# ans = RBF.get_y_regression(best, x_train, y_train, x_train)

# plots.scatter_plot(x_test[:, 0], x_test[:, 1], ans)
# plots.plot_regression_result(y_correct=y_train, y_model=ans)
