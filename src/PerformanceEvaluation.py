import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as sklm
from textwrap import wrap

def print_model_perf_stats(model, X, y):
    '''
    Prints stats relating to the model peformance of an estimator.

    Params:
        model  - estimator object which has been refit.
        X  - the test data.
        y - the actual labeles of the test set.
    '''

    y_predicted = model.predict(X)
    prfs = sklm.precision_recall_fscore_support(y, y_predicted)
    print("Evaluation Metrics: ")
    print("Accuracy     %0.3f" % sklm.accuracy_score(y, y_predicted))
    print("Macro precision      ", (sum(prfs[0]) / len(prfs[0])))
    print("Micro recall     ", (sum(prfs[1]) / len(prfs[1])))
    headings = ["Precision", "Recall", "F1", "Correct Predictions"]
    for i in range(len(prfs)):
        print("\n                     ", end='')
        for j in range(len(prfs[0])):
            print(j, "        ", end='')
        print('\n', headings[i], ":          ", end='')
        for j in range(len(prfs[0])):
            print(str(prfs[i][j]), "          ", end='')
        print()


def plot_rfecv_results(scores, reg, scoring, show=True, save=False, title_suffix='' ,path="plots/"):
    fig = plt.figure()
    ax = fig.gca()
    title = ax.set_title("\n".join(wrap("No. features vs classification " + str(scoring) + " L2 alpha = " + str(reg) + title_suffix, 60)))
    ax.set_xlabel("Number of features selected")
    ax.set_ylabel("Cross-validation " + scoring)
    plt.plot(range(1, len(scores) + 1), scores)
    plt.axvline(x=np.argmax(scores) + 1, linestyle='--')
    # plt.scatter(range(1, len(scores) + 1), scores)
    plt.grid(which='both')
    fig.tight_layout()
    title.set_y(1.05)
    fig.subplots_adjust(top=0.8)
    if save:
        plt.savefig(path.replace(' ', '_') + '_rfecv_results_reg_' + str(reg).replace('.', '_') + '_' + scoring + ".png", dpi=300)
    if show:
        plt.show()
    plt.close()


def plot_reg_vs_score(reg_strengths, training_acc, testing_acc, scoring, show=True, save=False, title_suffix='', path="plots/"):
    fig = plt.figure()
    ax = fig.gca()
    title = ax.set_title("\n".join(wrap("L2 Reg Strength vs " + str(scoring) + " for best performing model (post K-fold cv feature selection)" + title_suffix, 60)))
    ax.set_xlabel("Regularisation Strengths")
    ax.set_ylabel(scoring)
    # plt.plot(reg_strengths, training_acc, label="Training " + scoring, color="green")
    plt.scatter(reg_strengths, training_acc, label="Training " + scoring, color="green")
    # plt.plot(reg_strengths, testing_acc, label="Testing " + scoring, color="red")
    plt.scatter(reg_strengths, testing_acc, label="Testing " + scoring, color="red")
    plt.legend()
    plt.grid(which='both')
    fig.tight_layout()
    title.set_y(1.05)
    fig.subplots_adjust(top=0.8)
    if save:
        plt.savefig(path.replace(' ', '_') + '_reg_vs_' + scoring + '_results' + ".png", dpi=300)
    if show:
        plt.show()
    plt.close()

def plot_multiscore_comp(reg_list, y_lists, scorer_names, show=True, save=False, title_suffix='', path="plots/"):
    fig = plt.figure()
    ax = fig.gca()
    title = ax.set_title("\n".join(wrap("L2 Reg Strength vs testing scores for best peforming models (post K-fold cv feature selection)" + title_suffix, 60)))
    ax.set_xlabel("Regularisation Strengths")
    ax.set_ylabel("Score")
    for (list, scorer) in zip(y_lists, scorer_names):
        plt.plot(reg_list, list, '--', label=scorer)
    plt.legend(loc='best')
    plt.grid()
    fig.tight_layout()
    title.set_y(1.05)
    fig.subplots_adjust(top=0.8)
    if save:
        plt.savefig(path.replace(' ', '_') + "_scoring_comparison" + ".png", dpi=300)
    if show:
        plt.show()
    plt.close()


def plot_conf_matrix(y, y_predicted, classes, normalise=False, optimisation_strat='', title="Confusion Matrix", figsize=(10, 10), show=True, save=False, path="plots/"):
    '''
    This function plots a confusion matrix. It requires 3 inputs. y is the actual labels array. y_predicted is the predicted labels array. classes is an array of the classes. This code is adapted from: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html?fbclid=IwAR0fjfXZTMLc5_swKfZQut2-4bui0vgnqaT9atuZnSlo2HLOv9gnt_PEd0c
    '''
    if optimisation_strat != '':
        title += " (model optimisation: " + optimisation_strat + ')'
    if normalise:
        title = "Normalised " + title
    cm = sklm.confusion_matrix(y, y_predicted)
    if normalise:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=classes,
            yticklabels=classes,
            title=title,
            ylabel="Actual Label",
            xlabel="Predicted Label")
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    fmt = '.2f' if normalise else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    if save:
        plt.savefig(path.replace(' ', '_') + '_conf_matrix_model_optimisation_' + optimisation_strat + ".png", bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_grid_search_results(clf, std_thres = 1, show=True, save=False, path='plots/'):
    """
    Method to plot results from sklearn GridSearchCV function. Note, this methods only works where the same parameters are used throughout the grid search. For example, 'C' and 'kernel' in:
    params = [
        {'C': reg_params, 'kernel': ['linear']},
        {'C': reg_params, 'kernel': ['poly']},
        {'C': reg_params, 'kernel': ['rbf']},
        {'C': reg_params, 'kernel': ['sigmoid']}
    ]
    The first parameter will always be used as the independet variable (x-axis). In the above example that is 'C'. The other parameters will be brought in as sub-plots.

    Params:
        clf - the cv_results_ attribute of the fitted clf object.
        std_thres - standard deviation multiplier to use when shading std region
        show - boolean to show the plotted figure(s)
        save - boolean to save the figure(s)
        path - location to save the plotted figure
    """
    colors = ['red', 'blue', 'green', 'orange', 'magenta', 'gray', 'purple', 'brown', 'pink', 'black', 'yellow', 'cyan']

    # getting the parameter names used in the grid search
    params = list(clf.cv_results_['params'][0].keys())
    # getting the scorer names
    scorers = []
    for header in clf.cv_results_.keys():
        if "mean" in header:
            scorers += [header.replace('mean_', '')]
    # getting first param discreet vals
    replace_str = "bag__base_estimator__svc__"
    iv_vals = get_param_set(clf, params[0])
    for grid_param in params[1:]:
        for scorer in scorers:
            param_cats = get_param_set(clf, grid_param)
            fig = plt.figure()
            ax = fig.gca()
            y_lbl = scorer.replace(replace_str, '').replace('_', ' ')
            y_lbl = y_lbl.replace("test", "validation")
            x_lbl = params[0].replace(replace_str, '')
            title = ('Grid search cv results ' + x_lbl + ' vs ' + y_lbl + ' for different ' + grid_param.replace(replace_str, '') + 's')
            ax.set_title(title)
            ax.set_xlabel(x_lbl)
            ax.set_ylabel('Mean ' + y_lbl)
            i = 0
            for cat in param_cats:
                mu_list = np.array([])
                std_list = np.array([])
                for iv in iv_vals:
                    for param, mu, std in zip(clf.cv_results_['params'], clf.cv_results_['mean_' + scorer], clf.cv_results_['std_' + scorer]):
                        if param[params[0]] == iv and param[grid_param] == cat:
                            mu_list = np.append(mu_list, mu)
                            std_list =np.append(std_list, std)
                plt.plot(iv_vals, mu_list, '--', label=cat.replace(replace_str, '') + ' ' + grid_param.replace(replace_str, ''), color=colors[i])
                plt.plot(iv_vals, (mu_list + std_list * std_thres), '--', color=colors[i])
                plt.plot(iv_vals, (mu_list - std_list * std_thres), '--', color=colors[i])
                plt.fill_between(iv_vals, (mu_list + std_list * std_thres), (mu_list - std_list * std_thres), facecolor=colors[i], alpha=0.2)
                i += 1

            plt.legend(loc='best')
            plt.grid()
            if save:
                fig_name = (path + title.replace(' ', '_'))
                plt.savefig(fig_name + ".png", dpi=300)
            if show:
                plt.show()
            plt.close()


def get_param_set(clf, param):
    '''
    Helper method for plot_grid_search_results.
    '''
    param_cats = []
    for dict in clf.cv_results_['params']:
        param_cats += [dict[param]]
    temp = list(set(param_cats))
    temp.sort()
    return temp
