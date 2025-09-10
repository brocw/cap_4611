"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np

import utils
from utils import euclidean_dist_squared

import matplotlib.pyplot as plt


class KNN:
    X = None
    y = None

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X  # just memorize the training data
        self.y = y

    def predict(self, X_hat):
        mode = utils.mode(self.y)
        distance_arr = euclidean_dist_squared(self.X, X_hat)

        plt.plot(X_hat)
        plt.show()

        print(np.min(distance_arr[0]))

        for i, test_point in enumerate(X_hat):
            kth = np.partition(distance_arr[i], self.k)
            print(f"[{i}]: {kth[: self.k]}")
