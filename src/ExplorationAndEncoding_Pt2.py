"""
Author: Omar M Khalil - 170018405

This is the second part of ExplorationAndEncoding_pt1.py and includes further
changes to post-supervised datasets.

The file contains code used to:
1. Load the object containing the datasets.
2. Remove outliers within each class
3. Check for co-variate numerical features and reduce the features to remove
co-variance.
4. An algorithm is used to search for the best number of PCA reductions
5. Next a loop is entered iterating over the training and testing sets where:
    I. duplicathe data is dropped
    II. the data in all features is scaled from -1 to 1
    II. PCA, FAMD, and FAMD + Clustering is carried out the same datasets
        (clustering is not carried out on testing data)
    III. Plots to visualise training data are created
    IV. The PCA, FAMD, and FAMD-Clustered datasets are saved
"""

import pandas as pd
import pickle
from DataTransformation import *
from Plotting import *
from CleaningAndAugmentation import save_dataset
from FigureMate import FigureMate
from sklearn import preprocessing
from sklearn.decomposition import PCA


def main():
    fnames = ["trainingset_augmented_multiclass",
            "trainingset_augmented_binary",
            "testingset_augmented_multiclass",
            "testingset_augmented_binary"]

    title_prefix = ['Multiclass Training Set ',
        'Binary Training Set ',
        'Multiclass Testing Set ',
        'Binary Testing Set ']

    # loading label encodings
    mcl = pickle.load(open("../datasets/transformed/multiclass_label_encodings.obj", "rb"))
    bl = pickle.load(open("../datasets/transformed/binary_label_encodings.obj", "rb"))
    axes_labels = [mcl, bl]

    # # loading stored object containing datasets
    # print("*** Loading datasets.obj object ***")
    # datasets = pickle.load(
    #     open("../datasets/transformed/datasets_end_pt1.obj", "rb"))
    #
    # # first removing "outliers" datapoints 3 standard deviations from mean.
    # print("*** Removing outliers ***")
    # for i in range(2):
    #     datasets[i] = drop_outliers(datasets[i], datasets[i].iloc[:, 2:-1], 3, verbose=True)
    #
    # # next we can check to see if any numerical features are highly correlated
    # # (co-variate). This could allow us to reduce the number of features. The
    # # pearson r test is used to detect if there is any correlation between
    # # numerical features.
    # # the following code will find the correlated features for every class and
    # # append a list of tuples of the feature column numbers in
    # # all_correlated_features. Note, this process is only applied to the
    # # multiclass dataset since co-variate features in this dataset will also be
    # # covariate in the binary class, yet not neccessarily the other way around.
    # print("\n*** Checking for co-variate numerical features ***")
    # common_correlated_features = get_correlated_features(datasets[0],
    #     cols=datasets[0].columns[2:-1],
    #     classification=True,
    #     threshold=0.9,
    #     verbose=True)
    #
    # # now removing correlated features
    # print("removing correlated features")
    # for d in range(len(datasets)):
    #     for [i, j] in common_correlated_features:
    #         if j in datasets[d]:
    #             datasets[d] = datasets[d].drop([j], axis=1)
    # print("features remaining: ", datasets[0].columns)
    # print("number of features remaining: ", len(datasets[0].columns))

    # Step 4.
    datasets = pickle.load(open("../datasets/transformed/datasets_mid_pt2.obj", "rb"))
    # finding best number of components for PCA and FAMD reduction
    print("*** Finding optimal number of PCA and FAMD components ***")
    df_temp = balance_by_sampling(datasets[0]).sample(500)
    comp_PCA = get_best_reduction_PCA(df_temp,
                                      categotical_cols=df_temp.columns[:2],
                                      search_res=30,
                                      verbose=True,
                                      show=False,
                                      save=True,
                                      path="../plots/visualisation/after/")
    comp_FAMD = get_best_reduction_FAMD(df_temp,
                                        search_res=30,
                                        verbose=True,
                                        show=False,
                                        save=True,
                                        path="../plots/visualisation/after/")
    # Step 5.
    # initialising a pca, famd, and scaler
    pca = None
    famd = None
    scaler = preprocessing.MinMaxScaler().fit(datasets[0].iloc[:, :-1])
    # initialising lists of dfs to be saved
    pca_dfs = []
    famd_dfs = []
    famd_clustered_dfs = []
    trans_lbls = ["PCA", "FAMD", "FAMD-Clustered"]
    for i in range(len(datasets)):
        print("\n*** Carrying out unsupervised methods on %s ***" % fnames[i])
        df = datasets[i]

        # dropping duplicates
        df = df.drop_duplicates()

        # scaling numerical data
        scaled_data = scaler.transform(df.iloc[:, 2:-1].values)
        X = pd.DataFrame(scaled_data, index=df.index, columns=df.columns[2:-1])
        # re-joining categorical + numerical X and y
        df = pd.concat([df.iloc[:, :2], X, df.iloc[:, -1]], axis=1)

        # getting principal components using pca (numerical cols only!)
        print("*** Constructing PCA dataset ***")
        num_cols, pca = get_PCA_components(df.iloc[:, 2:],
                                         n_components=comp_PCA,
                                         fitted_pca_obj=pca)
        # encoding categorical data
        cat_cols = pd.get_dummies(df.iloc[:, :2], drop_first=True)
        # concatenating cat_cols and num_cols (note num_cols includes labels)
        df_pca = pd.concat([cat_cols, num_cols], axis=1)
        # adding df_pca to pca_dfs
        pca_dfs += df_pca

        # getting principal components using famd
        print("*** Constructing FAMD dataset ***")
        df_famd, famd = get_principal_components(df, comp_FAMD, famd_obj=famd)
        # adding df_famd to famd_dfs
        famd_dfs += df_famd

        # check if current dataframes are training sets
        if i < 2:
            # balancing sample counts for each class through clustering
            print("*** Constructing Clustered FAMD dataset ***")
            df_clustered = balance_sample_counts(df_famd,
                                                 max_clusters=3000,
                                                 mini_batch_multiplier=3,
                                                 verbose=True)
            famd_clustered_dfs += df_clustered

            for (df, lbl) in zip([df_pca, df_famd, df_clustered], trans_lbls):
                # plotting barchart to vis ditribution of labels
                print("Saving label count plot")
                fm = FigureMate(heading=title_prefix[i] + "Post " + lbl,
                                tick_labels=axes_labels[i%2],
                                path="../plots/visualisation/after/" + lbl)
                construct_frequency_plot(df, df.columns[-1], fm, show=0, save=1)

                # re-visualising seperabilitiy of left over engineered features
                print("\nSaving seprability plot")
                fm = FigureMate(heading=title_prefix[i%2] + "Post " + lbl,
                                legend_labels=axes_labels[i%2],
                                path="../plots/visualisation/after/" + lbl)
                construct_seperation_plot(df,
                                          df.columns[:-1],
                                          fm,
                                          std_dev=0.5,
                                          show=0,
                                          save=1)

                # visualising clusters
                print("\nSaving cluster plot")
                fm = FigureMate(heading=title_prefix[i%2] + "Post " + lbl,
                                legend_labels=axes_labels[i%2],
                                prefix=0,
                                path="../plots/visualisation/after/" + lbl)
                construct_cluster_plot(df,
                                       df.columns[:-1],
                                       fm,
                                       dimensions=3,
                                       show=0,
                                       save=1)

        for (df, lbl) in zip([df_pca, df_famd, df_clustered], trans_lbls):
            # saving final datasets
            print("\n*** Saving %s ***" % fnames[i])
            save_dataset(df,
                fnames[i] + "_" + lbl,
                "../datasets/transformed/postUnsupervised/",
                save_as="obj")

if __name__ == "__main__":
    main()
