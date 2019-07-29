import pickle
import sklearn.model_selection as ms
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from PerformanceEvaluation import *
from sklearn.metrics import make_scorer
from sklearn.feature_selection import RFECV
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import Pipeline

def get_selected_features(df, bool_list):
    kept_cols = []
    for i in range(len(bool_list)):
        if bool_list[i]:
            kept_cols += [df.columns[i]]
    return kept_cols


def train_and_test_svm(train, train_n, test, class_labels, stochastic=False, path_model="./", path_results="./"):
    """
    This method:
        1. Searches for the best possible SVM model using the passed kernel. The searches proccess involves a nested recursive feature removal across different regularisation intensities using stratified K fold cross-validation to optimise generalisation of the final model.
        2. Tests the final model and saves the results of its experiments.
    """
    X_trn = train.iloc[:, :-1]
    y_trn = train.iloc[:, -1]
    X_tst = test.iloc[:, :-1]
    y_tst = test.iloc[:, -1]

    scoring = {'precision': make_scorer(precision_score, average='weighted'),
                'recall': make_scorer(recall_score, average='weighted'),
                'f1_score': make_scorer(f1_score, average='weighted')}
    reg_strengths = None
    if stochastic:
        reg_strengths = np.arange(0.001, 0.99, 0.05).round(decimals=3)
    else:
        reg_strengths = np.arange(0.1, 1, 0.1).round(decimals=3)
    max_test_scores = []
    rfecv_lists = []
    best_model_indexes = []
    for scorer in scoring:
        max_train_score = []
        max_test_score = []
        rfecv_list = []
        for c in reg_strengths:
            # creating classifer
            svc = None
            if stochastic:
                svc = SGDClassifier(loss="hinge", penalty="l2", alpha=c, max_iter=10000, tol=1e-5, n_jobs=-1, learning_rate="adaptive", early_stopping=True, class_weight="balanced", eta0=1)
            else:
                svc = LinearSVC(C=c, class_weight="balanced", max_iter=10000)

            # creating recursive feature elimination model with cross validation
            rfecv = RFECV(
                svc,
                step=1,
                cv=ms.StratifiedKFold(10),
                scoring=scorer,
                n_jobs=-1)

            print("training set size: ", len(X_trn), " testing set size: ", len(X_tst))
            print("*** Beginning training ***")
            rfecv.fit(X_trn, y_trn)
            rfecv_list += [rfecv]
            # getting scores for predicting on the training and testing datasets
            max_train_score += [rfecv.score(X_trn, y_trn)]
            max_test_score += [rfecv.score(X_tst, y_tst)]

        max_test_scores += [max_test_score]
        rfecv_lists += [rfecv_list]

        # plotting results
        best_model_index = max_test_score.index(max(max_test_score))
        best_model_indexes += [best_model_index]
        print("Optimal number of features: ", rfecv.n_features_)
        plot_rfecv_results(rfecv.grid_scores_, reg_strengths[best_model_index], scorer, show=False, save=True, path=path_results + train_n + '_')
        plot_reg_vs_score(reg_strengths, max_train_score, max_test_score, scorer, show=False, save=True, path=path_results + train_n + '_')


        best_model = rfecv_list[best_model_index]
        print_model_perf_stats(best_model, X_tst, y_tst)
        plot_conf_matrix(y_tst, best_model.predict(X_tst), class_labels, optimisation_strat=scorer, normalise=True, show=False, save=True, path=path_results + train_n + '_')

        # saving useful column names
        selected_features = get_selected_features(X_trn, rfecv.support_)
        print("Features kept with %s as the optimisation strategy: " % scorer)
        print(selected_features)
        pickle.dump(selected_features, open(path_results + train_n + "_usefulFeatures_model_optimisation_strat_" + scorer, "wb"))

    # making plot combining all scorers vs reg on a single figure
    plot_multiscore_comp(reg_strengths, max_test_scores, list(scoring), show=False, save=True, path=path_results + train_n + '_')

    # saving trained models
    pickle.dump(rfecv_lists, open(path_model + fname + "_trained_models.obj", "wb"))
