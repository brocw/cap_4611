#!/usr/bin/env python
import argparse
import os
import sys
import pickle
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

# our code
from utils import load_dataset, plot_classifier, handle, run, main
from decision_stump import DecisionStumpInfoGain
from decision_tree import DecisionTree
from kmeans import Kmeans
from knn import KNN
from naive_bayes import NaiveBayes, NaiveBayesLaplace
from random_tree import RandomForest, RandomTree


@handle("1")
def q1():
    dataset = load_dataset("citiesSmall.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]

    kk = [1, 3, 10]

    for i in kk:
        k = i
        model = KNN(k)
        model.fit(X, y)

        y_hat = model.predict(X)
        err_train = np.mean(y_hat != y)

        y_pred = model.predict(X_test)
        err_test = np.mean(y_pred != y_test)

        plot_classifier(model, X_test, y_test)

        print(f"k={i}    Training error: {err_train:.3f}")
        print(f"k={i}    Testing error: {err_test:.3f}")


@handle("2")
def q2():
    dataset = load_dataset("ccdebt.pkl")
    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]

    ks = list(range(1, 30, 4))
    ks_graph = np.zeros((len(ks), 10))
    cv_graph = []
    test_graph = []
    train_graph = []
    for index, kk in enumerate(ks):
        k = kk

        # 10-fold cross-validation
        cv_accs = 0.0
        n = X.shape[0]
        for fold in range(10):
            # Calculating partition size
            # From: https://g.co/gemini/share/1b623820fd6f
            mask_arr = np.ones(n, dtype=int)
            base_size, remainder = divmod(n, 10)
            start_index = fold * base_size + min(fold, remainder)
            partition_size = base_size + 1 if fold < remainder else base_size
            end_index = start_index + partition_size

            mask_arr[start_index:end_index] = 0
            train_mask = mask_arr.astype(bool)
            test_mask = ~train_mask

            cross_test = X[test_mask]
            cross_y_test = y[test_mask]
            cross_train = X[train_mask]
            cross_y_train = y[train_mask]

            model = KNN(k)
            model.fit(cross_train, cross_y_train)

            y_pred = model.predict(cross_test)
            fold_err_test = np.mean(y_pred != cross_y_test)

            print(f"k={k} fold={fold}          Testing error: {fold_err_test:.3f}")
            ks_graph[index][fold] = fold_err_test

            cv_accs += fold_err_test

        cv_accs /= 10
        cv_graph.append(1 - cv_accs)

        # Test accuracy
        y_hat = model.predict(X)
        err_train = np.mean(y_hat != y)
        train_graph.append(err_train)
        y_pred = model.predict(X_test)
        err_test = np.mean(y_pred != y_test)
        test_graph.append(1 - err_test)

    k_values = [
        "1",
        "5",
        "9",
        "13",
        "17",
        "21",
        "25",
        "29",
    ]

    k_values_int = [int(k) for k in k_values]

    # fname = Path("..", "figs", "k2_graph.png")
    # fig, ax = plt.subplots()
    # ax.plot(k_values_int, cv_graph, marker="o", label="Cross-Validation Accuracy")
    # ax.plot(k_values_int, test_graph, marker="s", label="Test Accuracy")
    # ax.legend()
    # plt.xlabel("K-Values")
    # ax.set_xticks(k_values_int)
    # plt.ylabel("Accuracy")
    # plt.title("Cross-Validation & Test Accuracy Comparison")
    # plt.savefig(fname)

    f2name = Path("..", "figs", "k2_error_graph.png")
    plt.plot(k_values, train_graph, marker="o")
    plt.xlabel("K-Values")
    plt.ylabel("Accuracy")
    plt.title("Training error VS. K")
    plt.savefig(f2name)


@handle("3.2")
def q3_2():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"].astype(bool)
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]
    groupnames = dataset["groupnames"]
    wordlist = dataset["wordlist"]

    print(wordlist[72])
    print(X[802])
    for i, sample in enumerate(X[802]):
        if sample:
            print(wordlist[i])

    print(groupnames)
    print(groupnames[y[802]])


@handle("3.3")
def q3_3():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]

    print(f"d = {X.shape[1]}")
    print(f"n = {X.shape[0]}")
    print(f"t = {X_valid.shape[0]}")
    print(f"Num classes = {len(np.unique(y))}")

    model = NaiveBayes(num_classes=4)
    model.fit(X, y)

    y_hat = model.predict(X)
    err_train = np.mean(y_hat != y)
    print(f"Naive Bayes training error: {err_train:.3f}")

    y_hat = model.predict(X_valid)
    err_valid = np.mean(y_hat != y_valid)
    print(f"Naive Bayes validation error: {err_valid:.3f}")


@handle("3.4")
def q3_4():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]

    print(f"d = {X.shape[1]}")
    print(f"n = {X.shape[0]}")
    print(f"t = {X_valid.shape[0]}")
    print(f"Num classes = {len(np.unique(y))}")

    model = NaiveBayes(num_classes=4)
    model.fit(X, y)

    y_hat = model.predict(X)
    err_train = np.mean(y_hat != y)
    print(f"Naive Bayes training error: {err_train:.3f}")

    y_hat = model.predict(X_valid)
    err_valid = np.mean(y_hat != y_valid)
    print(f"Naive Bayes validation error: {err_valid:.3f}")

    model_laplace = NaiveBayesLaplace(num_classes=4, beta=1)
    model_laplace.fit(X, y)

    y_hat = model_laplace.predict(X)
    err_train = np.mean(y_hat != y)
    print(f"Laplace Naive Bayes training error: {err_train:.3f}")

    y_hat = model_laplace.predict(X_valid)
    err_valid = np.mean(y_hat != y_valid)
    print(f"Laplace Naive Bayes validation error: {err_valid:.3f}")

    print("beta = 10000")
    model_laplace_big_beta = NaiveBayesLaplace(num_classes=4, beta=10000)
    model_laplace_big_beta.fit(X, y)

    y_hat = model_laplace_big_beta.predict(X)
    err_train = np.mean(y_hat != y)
    print(f"Laplace Naive Bayes training error: {err_train:.3f}")

    y_hat = model_laplace_big_beta.predict(X_valid)
    err_valid = np.mean(y_hat != y_valid)
    print(f"Laplace Naive Bayes validation error: {err_valid:.3f}")

    print(f"Naive p value: {model.p_xy[1][1]}")
    print(f"Laplace p value: {model_laplace.p_xy[1][1]}")
    print(f"Big Laplace p value: {model_laplace_big_beta.p_xy[1][1]}")


@handle("4")
def q4():
    dataset = load_dataset("vowel.pkl")
    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]
    print(f"n = {X.shape[0]}, d = {X.shape[1]}")

    def evaluate_model(model):
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print(f"    Training error: {tr_error:.3f}")
        print(f"    Testing error: {te_error:.3f}")

    print("Decision tree info gain")
    evaluate_model(DecisionTree(max_depth=np.inf))

    print("Random tree info gain")
    evaluate_model(RandomTree(max_depth=np.inf))

    print("Random Forest info gain")
    evaluate_model(RandomForest(num_trees=50, max_depth=np.inf))


@handle("5")
def q5():
    X = load_dataset("clusterData.pkl")["X"]

    model = Kmeans(k=4)
    model.fit(X)
    y = model.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="jet")

    fname = Path("..", "figs", "kmeans_basic_rerun.png")
    plt.savefig(fname)
    print(f"Figure saved as {fname}")


@handle("5.1")
def q5_1():
    X = load_dataset("clusterData.pkl")["X"]

    best_error = 9999999
    fname = Path("..", "figs", "kmeans_5_1_lowest_error.png")
    for i in range(50):
        model = Kmeans(k=4)
        model.fit(X)
        y = model.predict(X)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap="jet")
        error = model.error(X, y, model.means)
        print(f"[{i}]: Error {error}")
        if error < best_error:
            best_error = error
            plt.scatter(X[:, 0], X[:, 1], c=y, cmap="jet")
            plt.savefig(fname)
            print("Best error so far, saved.")


@handle("5.2")
def q5_2():
    X = load_dataset("clusterData.pkl")["X"]

    fname = Path("..", "figs", "kmeans_5_2_lowest_error.png")
    k_graph = [
        999999,
        999999,
        999999,
        999999,
        999999,
        999999,
        999999,
        999999,
        999999,
        999999,
    ]
    for i in range(50):
        k = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        model = Kmeans(k)
        model.fit(X)
        y = model.predict(X)
        error = model.error(X, y, model.means)
        # k_graph.append((k, error))
        print(f"[{i}] k={k}: Error {error}")
        if error < k_graph[k - 1]:
            k_graph[k - 1] = error
            print(f"      k={k} Best error so far, saved.")

    plt.plot(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"], k_graph, marker="o")
    plt.xlabel("K-Values")
    plt.ylabel("Error")
    plt.savefig(fname)


if __name__ == "__main__":
    main()
