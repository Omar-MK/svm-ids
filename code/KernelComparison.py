"""
Author: Omar M Khalil 170018405

In this file different SVM kernels are compared for their effectiveness at seperating the intrusion datasets.

1. The cleaned and encoded (augmented) multiclass, pre and post-supervised datasets are loaded.
2. Using a grid search cross validation algorithm, the performance of different kernels is compared at different regularisation intensities.
"""

import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import Pipeline
from sklearn import datasets
from DataTransformation import balance_by_sampling
from Plotting import *
from PerformanceEvaluation import *


def main():
    # loading datasets
    print("*** Loading datasets ***")
    path1 = "../datasets/transformed/preUnsupervised/"
    path2 = "../datasets/transformed/postUnsupervised/"

    train = pickle.load(open(path1 + "trainingset_augmented_multiclass.obj", "rb"))

    test = pickle.load(open(path1 + "testingset_augmented_multiclass.obj", "rb"))
    train_u = pickle.load(open(path2 + "trainingset_augmented_multiclass_unsupervised.obj", "rb"))

    test_u = pickle.load(open(path2 + "testingset_augmented_multiclass_unsupervised.obj", "rb"))
    # loading class label encodings
    multiclass_labels = pickle.load(open("../datasets/transformed/multiclass_label_encodings.obj", "rb"))
    # balancing training set samples and randomly sampling result
    train = balance_by_sampling(train)
    train = train.sample(500)
    # post-supervised training set already balanced
    train_u = train_u.sample(500)

    print("*** Initialising training process ***")
    # grid search kernel and regularisation params
    reg_params = np.arange(0.1, 2, 0.1)
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    params = [
        {'bag__base_estimator__svc__C': reg_params, 'bag__base_estimator__svc__kernel': kernels}
    ]

    # scoring functions
    scoring = {'precision': make_scorer(precision_score, average='weighted'),
        'recall': make_scorer(recall_score, average='weighted'),
        'f1_score': make_scorer(f1_score, average='weighted')}

    # svm object, gamma specifies kernel coefficent for rbf, poly, and sigmoid.
    # default scale =  1 / (n_features * X.var())
    svc_pipe = Pipeline([(
        "svc",
        SVC(gamma="scale",
        class_weight="balanced",
        probability=True,
        cache_size=200))])

    pipe = Pipeline([("bag", BaggingClassifier(svc_pipe, n_estimators=10))])
    fig_markers = ["pre_unsupervised", "post_unsupervised"]

    print("*** Training and Testing ***")
    # now iterating over both datasets
    i = 0
    for (trn, tst) in [(train, test), (train_u, test_u)]:
        print("training set size: ", len(trn), " testing set size: ", len(tst))

        # creating grid search object.
        # n_jobs number of cpu cores to use (-1 is all)
        # pre_dispatch - controls the number of jobs that get dispatched during
        # execution. By default this uses a stratified k fold cross-validation
        # splitting strategy as long as a number is passed in cv.
        clf = GridSearchCV(
            pipe,
            params,
            refit='f1_score',
            cv=10,
            scoring=scoring,
            n_jobs=-1,
            pre_dispatch='n_jobs')


        # fitting over training data
        clf.fit(trn.iloc[:, :-1], trn.iloc[:, -1])
        # now plotting grid search results on held
        print("The best score was: ", clf.best_score_)
        print("The best parameters were: ", clf.best_params_)
        plot_grid_search_results(
            clf,
            std_thres = 0.5,
            show=False,
            save=True,
            path="../plots/results/kernelComp/" + fig_markers[i%2])

        # plotting confusion matrix on test of best f1 score model
        print_model_perf_stats(clf, tst.iloc[:, :-1], tst.iloc[:, -1])
        plot_conf_matrix(
            tst.iloc[:, -1],
            clf.predict(tst.iloc[:, :-1]),
            classes=multiclass_labels,
            optimisation_strat="F1 score",
            normalise=True,
            show=False,
            save=True,
            path="../plots/results/kernelComp/" + fig_markers[i%2])
        i += 1

if __name__ == "__main__":
    main()
