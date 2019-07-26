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
from DataTransformation import *
from Plotting import *
from PerformanceEvaluation import *

def load_datasets(files_names):
    data = []
    for fn in files_names:
        data += [pickle.load(open(fn + ".obj", "rb"))]
    return data

def main():
    # loading datasets
    print("*** Loading datasets ***")
    path1 = "../datasets/transformed/preUnsupervised/"
    path2 = "../datasets/transformed/postUnsupervised/"
    train_prefix = "trainingset_augmented_multiclass"
    fig_markers = ["pre-unsupervised", "post-unsupervised_unclustered", "post-unsupervised_clustered", "post-unsupervised_PCA"]
    train_files = [path1 + train_prefix ,
                   path2 + train_prefix + "_unclustered_unsupervised",
                   path2 + train_prefix + "_unsupervised",
                   path2 + train_prefix + "_pca_unsupervised"]
    datasets = load_datasets(train_files)

    # loading class label encodings
    labels = pickle.load(open("../datasets/transformed/multiclass_label_encodings.obj", "rb"))
    trn_sets = []
    tst_sets =  []
    for df in datasets:
        # splitting data into training and testing sets
        X_trn, X_tst, y_trn, y_tst = split_train_test(df.iloc[:, :-1], df.iloc[:, -1] , 0.2)
        trn_set = pd.concat([X_trn, y_trn], axis=1)
        tst_set = pd.concat([X_tst, y_tst], axis=1)
        # balancing training set samples and randomly sampling result
        trn_set = balance_by_sampling(trn_set)
        trn_set = trn_set.sample(1000)
        trn_sets += [trn_set]
        tst_sets += [tst_set]


    print("*** Initialising training process ***")
    # grid search kernel and regularisation params
    reg_params = np.arange(0.1, 0.2, 0.1)
    kernels = ['linear'] #, 'poly', 'rbf', 'sigmoid']
    params = [{'bag__base_estimator__svc__C': reg_params,
               'bag__base_estimator__svc__kernel': kernels}]

    # scoring functions
    scoring = {'precision': make_scorer(precision_score, average='weighted'),
               'recall': make_scorer(recall_score, average='weighted'),
               'f1_score': make_scorer(f1_score, average='weighted')}

    # svm object, gamma specifies kernel coefficent for rbf, poly, and sigmoid.
    # default scale =  1 / (n_features * X.var())
    svc_pipe = Pipeline([("svc",
                          SVC(gamma="scale",
                          class_weight="balanced",
                          probability=True,
                          cache_size=200))])

    pipe = Pipeline([("bag", BaggingClassifier(svc_pipe, n_estimators=4))])

    print("*** Training and Testing ***")
    # now iterating over both datasets
    i = 0
    for (trn, tst, file) in zip(trn_sets, tst_sets, train_files):
        print("training set size: ", len(trn), " testing set size: ", len(tst))

        # creating grid search object.
        # n_jobs number of cpu cores to use (-1 is all)
        # pre_dispatch - controls the number of jobs that get dispatched during
        # execution. By default this uses a stratified k fold cross-validation
        # splitting strategy as long as a number is passed in cv.
        clf = GridSearchCV(pipe,
                           params,
                           refit='f1_score',
                           cv=10,
                           scoring=scoring,
                           n_jobs=-1,
                           pre_dispatch='n_jobs',
                           verbose=True)


        # fitting over training data
        clf.fit(trn.iloc[:, :-1], trn.iloc[:, -1])
        # now plotting grid search results on held
        print("The best score was: ", clf.best_score_)
        print("The best parameters were: ", clf.best_params_)
        plot_grid_search_results(clf,
                                 std_thres = 0.5,
                                 show=False,
                                 save=True,
                                 path="../plots/results/kernelComp/" + fig_markers[i])

        # plotting confusion matrix on test of best f1 score model
        # print_model_perf_stats(clf, tst.iloc[:, :-1], tst.iloc[:, -1])
        plot_conf_matrix(tst.iloc[:, -1],
                         clf.predict(tst.iloc[:, :-1]),
                         classes=labels,
                         title="Confusion Matrix " + fig_markers[i].replace('_', ''),
                         normalise=True,
                         show=False,
                         save=True,
                         path="../plots/results/kernelComp/" + fig_markers[i])
        i += 1

if __name__ == "__main__":
    main()
