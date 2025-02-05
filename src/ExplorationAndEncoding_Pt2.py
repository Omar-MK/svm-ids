"""
Author: Omar M Khalil - 170018405

This is the second part of the exploration process.
The main objective of this file is to explore which component analysis reduction technique is best to reduce the datasets.

The file contains code used to:
1. Load the object containing the datasets.
2. Check for co-variate numerical features and reduce the features to remove
co-variance.
3. Grid search is used to search for the best number of component analysis
reductions and max score achieved using:
    I. PCA applied on the numerical columns of the dataset,
    II. PCA applied on both the numerical and dummy encoded categorical columns,
    III. FAMD applied on both the numerical and categorical columns.
4. The datasets leading to the best score are saved for use in
   ExplorationAndEncoding_pt3.
"""

import pickle
from DataTransformation import *
import statistics as stats


def choose_features_to_drop(df, correlated_features):
    """
    Out of a tuple of correlated features (columns) in a dataframe, chooses to
    keep a feature by calculating the mean value of the current column for
    each class, the mean difference between each adjacent mean value is then
    calculated and this is used to select the column which seperates the
    classes more optimally.
    Note the last column should be the classes column.
    """
    to_drop = []
    for (i, j) in correlated_features:
        i_class_means = []
        j_class_means = []
        i_class_diffs = []
        j_class_diffs = []
        # for each label work out mean ith col and jth col value
        for label in set(df.iloc[:, -1]):
            df_curr = df[df.Label == label]
            i_class_means += [df_curr[i].mean()]
            j_class_means += [df_curr[j].mean()]
        # now sorting and calculating averages differences between adjacents
        i_class_means.sort()
        j_class_means.sort()
        for n in range(1, len(i_class_means)):
            i_class_diffs += [abs(i_class_means[n] - i_class_means[n - 1])]
            j_class_diffs += [abs(j_class_means[n] - j_class_means[n - 1])]
        # adding col to drop in to_drop based on mean diffs
        if stats.mean(i_class_diffs) > stats.mean(j_class_diffs):
            to_drop += [j]
        else:
            to_drop += [i]
    return to_drop


def main():
    print("*** Loading datasets object ***")
    # loading stored object containing datasets to be used for PCA (numerical
    # cols only) and FAMD
    datasets = pickle.load(open("../datasets/transformed/datasets_end_pt1.obj", "rb"))
    # loading stored object containing datasets to be used for PCA containing
    # dummy encoded categorical features.
    datasets_encoded = pickle.load(open("../datasets/transformed/datasets_end_pt1_cat_encoded.obj", "rb"))

    # next we can check to see if any numerical features are highly correlated
    # (co-variate). This could allow us to reduce the number of features. The
    # pearson r test is used to detect if there is any correlation between
    # numerical features.
    # the following code will find the correlated features for every class and
    # append a list of tuples of the feature column numbers in
    # all_correlated_features. Note, this process is only applied to the
    # multiclass dataset since co-variate features in this dataset will also be
    # covariate in the binary class, yet not neccessarily the other way around.
    print("\n*** Checking for co-variate numerical features ***")
    corr_features = get_correlated_features(datasets[0],
                                            cols=datasets[0].columns[2:-1],
                                            classification=True,
                                            threshold=0.9,
                                            verbose=True)
    cols_to_drop = choose_features_to_drop(datasets[0], corr_features)
    # now removing correlated features from all datasets
    print("removing correlated features")
    for n in range(len(datasets)):
        datasets[n] = datasets[n].drop(cols_to_drop, axis=1)
        datasets_encoded[n] = datasets_encoded[n].drop(cols_to_drop, axis=1)

    print("features remaining: ", datasets[0].columns)
    print("number of features remaining: ", len(datasets[0].columns))

    # Step 4.
    # finding best number of components for PCA and FAMD reductions
    print("*** Finding optimal number of PCA and FAMD components ***")
    df_temp = balance_by_sampling(datasets[0]).sample(1000)
    df_temp_with_cat = balance_by_sampling(datasets_encoded[0]).sample(1000)
    PCA_res = get_best_reduction_PCA(df_temp,
                                     categotical_cols=df_temp.columns[:2],
                                     search_res=30,
                                     verbose=True,
                                     show=False,
                                     save=True,
                                     path="../plots/visualisation/after/")
    FAMD_res = get_best_reduction_FAMD(df_temp,
                                       search_res=30,
                                       verbose=True,
                                       show=False,
                                       save=True,
                                       path="../plots/visualisation/after/")
    PCA_with_cat_res = get_best_reduction_PCA(df_temp_with_cat,
                                              search_res=30,
                                              verbose=True,
                                              show=False,
                                              save=True,
                                              path="../plots/visualisation/after/with_cat_")

    if max(PCA_res[1], FAMD_res[1]) > PCA_with_cat_res[1]:
        pickle.dump(datasets, open("../datasets/transformed/datasets_end_pt2.obj", "wb"))
        if PCA_res[1] > FAMD_res[1]:
            print("PCA carried out on the numerical columns only was the best technique")
            pickle.dump(PCA_res[0], open("../datasets/transformed/n_components.obj", "wb"))
        else:
            print("FAMD was the best techniqe")
            pickle.dump(FAMD_res[0], open("../datasets/transformed/n_components.obj", "wb"))
    else:
        pickle.dump(datasets_encoded, open("../datasets/transformed/datasets_end_pt2.obj", "wb"))
        print("PCA carried out on both the numerical and dummy encoded categorical features was the best technique")
        pickle.dump(PCA_with_cat_res[0], open("../datasets/transformed/n_components.obj", "wb"))


if __name__ == "__main__":
    main()
