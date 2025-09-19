from random_stump import RandomStumpInfoGain
from decision_tree import DecisionTree
import numpy as np

import utils

import random


class RandomTree(DecisionTree):
    def __init__(self, max_depth):
        DecisionTree.__init__(
            self, max_depth=max_depth, stump_class=RandomStumpInfoGain
        )

    def fit(self, X, y):
        n = X.shape[0]
        boostrap_inds = np.random.choice(n, n, replace=True)
        bootstrap_X = X[boostrap_inds]
        bootstrap_y = y[boostrap_inds]

        DecisionTree.fit(self, bootstrap_X, bootstrap_y)


class RandomForest:
    num_trees = None
    max_depth = None

    models = []

    def __init__(self, num_trees, max_depth):
        self.num_trees = num_trees
        self.max_depth = max_depth

    def fit(self, X, y):
        n = X[0].size

        for t in range(self.num_trees):
            # Bootstrapping data
            X_boot = X.copy()
            y_boot = y.copy()
            for i in range(n):
                j = np.random.randint(n, size=1)[0] + 1
                X_boot[i, :] = X[j, :]
                y_boot[i] = y[j]

    def predict(self, X_pred):
        n, d = X_pred.shape
        y = np.zeros(n)

        # Get predictions
        y_hat_trees = np.empty((n, self.num_trees))
        for t in range(self.num_trees):
            predictions = models[t].predict(X_pred)
            y_hat_trees[:, t] = predictions

        # Find mode of predictions, add to y
        for x in range(n):
            pred_array = y_hat_trees[:, x]
            y[x] = utils.mode(pred_array)

        return y
