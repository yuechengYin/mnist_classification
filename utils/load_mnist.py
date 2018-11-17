import os
import gzip
import pickle
import wget

import numpy as np


def load_mnist_data():

    data_file = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(data_file,encoding='bytes')
    data_file.close()

    X_train = training_data[0].reshape(50000, 1, 28, 28)
    Y_train = training_data[1]
    training_data = (X_train, Y_train)

    X_val = validation_data[0].reshape(10000, 1, 28, 28)
    Y_val = validation_data[1]
    validation_data = (X_val, Y_val)

    X_test = test_data[0].reshape(10000, 1, 28, 28)
    Y_test = test_data[1]
    test_data = (X_test, Y_test)
    return training_data, validation_data, test_data

'''
def vectorized_result(y):
    e = np.zeros((10, 1))
    e[y] = 1.0
    return e
'''
