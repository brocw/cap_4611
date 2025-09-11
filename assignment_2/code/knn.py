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
        y_hat = np.arange(X_hat.shape[0])
        distance_arr = euclidean_dist_squared(self.X, X_hat)

        # For each test point
        for i, test_point in enumerate(X_hat):
            # Tuple array (distance, class) for all training points
            training_points = []
            for j, training_point in enumerate(self.X):
                point_class = self.y[j]
                training_points.append((distance_arr[j, i], point_class))

            # Sorting by distance, kth closest points
            training_points.sort()
            kth_points = training_points[: self.k]

            # Getting nearest labels for kth closest points, finding prediction
            k_nearest_labels = [point[1] for point in kth_points]
            numpy_k_nearest_labels = np.asarray(k_nearest_labels)
            prediction = utils.mode(numpy_k_nearest_labels)

            # Ties for even split of nearest labels
            if (
                numpy_k_nearest_labels.size % 2 == 0
                and prediction == numpy_k_nearest_labels.size / 2
            ):
                prediction = mode
            y_hat[i] = prediction

        return y_hat
