import numpy as np


def random_classification_data(number_of_records, dim, number_of_classes):
    x_data = np.random.rand(number_of_records, dim)
    y_data = np.random.rand(number_of_records, 1)
    y_data = number_of_classes * y_data
    y_data = y_data.astype(int)
    return x_data, y_data


def classification_data(number_of_records, dim, number_of_classes):
    recs = int(number_of_records / number_of_classes)
    data = []
    for i in range(number_of_classes - 1):
        data.append(np.random.normal(5 * i, 1, size=(recs, dim)))
        # data2 = np.random.normal(12, 1, size=(re, dim))
    data.append(np.random.normal(5 * (number_of_classes - 1), 1,
                                 size=(number_of_records - (number_of_classes - 1) * recs, dim)))
    y = []
    for i in range(number_of_classes - 1):
        y.append((i) * np.ones(shape=(recs)))
    # y2 = 2 * np.ones(shape=(number_of_records, 1))
    y.append((number_of_classes - 1) * np.ones(shape=(number_of_records - (number_of_classes - 1) * recs)))
    # print(np.ones(shape=(number_of_records - (number_of_classes - 1) * recs)))
    # print(len(y))
    # print(tuple(i for i in y))
    x_data = np.concatenate(tuple(i for i in data), axis=0)
    y_data = np.concatenate(tuple(i for i in y), axis=0)
    y_data = y_data.astype(int)

    return x_data, y_data


def classification_data2(number_of_records, dim, number_of_classes):
    from sklearn.datasets import make_gaussian_quantiles
    # Construct dataset
    # Gaussian 1
    recs = int(number_of_records/2)
    X1, y1 = make_gaussian_quantiles(cov=3.,
                                     n_samples=recs, n_features=dim,
                                     n_classes=number_of_classes, random_state=1)

    # Gaussian 2
    X2, y2 = make_gaussian_quantiles(mean=(4, 4), cov=1,
                                     n_samples=recs, n_features=dim,
                                     n_classes=number_of_classes, random_state=1)
    x = np.concatenate((X1, X2), axis=0)
    y = np.concatenate((y1, y2), axis=0)
    return x, y


def regression_data(number_of_records, dim):
    x_data = np.random.rand(number_of_records, dim)
    y_data = np.random.rand(number_of_records, 1)
    return x_data, y_data

#
# def normalize_data(data):
#     mu = np.mean(data, axis=0)
#     print(mu)
#     print(np.reshape(mu, (1, -1)))
#     print(np.shape(data))
#     sigma = np.std(data, axis=0)
#     data = data - np.matmul(np.ones(shape=(len(data), 1), dtype=int), np.reshape(mu, (1, -1)))
#     print(sigma)
#     mat = np.zeros(shape=(len(sigma), len(sigma)))
#     mat[0] = np.reciprocal(sigma)
#     data = np.matmul(data, mat)
#     return data, mu, sigma
