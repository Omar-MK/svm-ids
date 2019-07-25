"""
Author: Omar M Khalil - 170018405

This is the second part of ExplorationAndEncoding_pt1.py and includes further
changes to post-supervised datasets.

The file contains code used to:
1. Load the object containing the datasets.
2. Remove outliers within each class
3. Check for co-variate numerical features and reduce the features to remove
co-variance.
4. carry out FAMD projection of the data to project the features into a lower
feature space. This has the effect of turning all features into numeric
features.
5. Using the numeric feautres, KMeans clustering is applied balance data point
count per class.
6. Re-check for outliers, and based on user input, the outliers are removed or
kept.
7. Plot the seprability plot and 3D cluster plots from steps 10 and 11 are
remade to visualise effect of changes.
8. Save the post-superivsed datasets.
"""

import pandas as pd
import pickle
from DataTransformation import *
from Plotting import *
from CleaningAndAugmentation import save_dataset
from FigureMate import FigureMate
from sklearn import preprocessing

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

    # loading stored object containing datasets
    print("*** Loading datasets.obj object ***")
    datasets = pickle.load(
        open("../datasets/transformed/datasets_end_pt1.obj", "rb"))

    # first removing "outliers" datapoints 3 standard deviations from mean.
    print("*** Removing outliers ***")
    for i in range(2):
        datasets[i] = drop_outliers(datasets[i], datasets[i].iloc[:, 2:-1], 3, verbose=True)

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
    common_correlated_features = get_correlated_features(datasets[0],
        cols=datasets[0].columns[2:-1],
        classification=True,
        threshold=0.9,
        verbose=True)

    # now removing correlated features
    print("removing correlated features")
    for d in range(len(datasets)):
        for [i, j] in common_correlated_features:
            if j in datasets[d]:
                datasets[d] = datasets[d].drop([j], axis=1)
    print("features remaining: ", datasets[0].columns)
    print("number of features remaining: ", len(datasets[0].columns))

    # the next step is to peform principal component analysis (FAMD) to project
    # and cat features and numerical features onto common planes. This also
    # reduces the dimensionality of the datasets. This is followed by KMeans
    # clustering to balance out the counts of rows for each class.

    print("\n*** Carrying out FAMD and clustering ***")
    multiclass_scaler = preprocessing.MinMaxScaler()
    binary_scaler = preprocessing.MinMaxScaler()
    for i in range(len(datasets)):
        datasets[i] = datasets[i].drop_duplicates()
        df = datasets[i]
        # getting principal components
        scaled_features = []
        df = get_principal_components(df, len(datasets[0].columns) - 11)
        if i == 0:
            multiclass_scaler.fit(df.iloc[:, :-1])
        elif i == 1:
            binary_scaler.fit(df.iloc[:, :-1])
        if i % 2 == 0:
            scaled_features = multiclass_scaler.transform(df.iloc[:, :-1].values)
        else:
            scaled_features = binary_scaler.transform(df.iloc[:, :-1].values)
        df_X = pd.DataFrame(scaled_features, index=df.index, columns=df.columns[:-1])
        df = pd.concat([df_X, df.iloc[:, -1]], axis=1)
        if i < 2:
            # balancing sample counts for each class through clustering - only
            # applied on training sets
            print("now clustering data")
            df = balance_sample_counts(df, max_clusters=3000, mini_batch_multiplier=3, verbose=True)

            # plotting barchart to vis ditribution of labels
            print("Saving label count plot")
            fm = FigureMate(heading=title_prefix[i], tick_labels=axes_labels[i%2], path="../plots/visualisation/after/")
            construct_frequency_plot(df, df.columns[-1], fm, show=0, save=1)

            # re-visualising seperabilitiy of left over engineered features
            print("\nSaving seprability plot")
            fm = FigureMate(heading=title_prefix[i%2], legend_labels=axes_labels[i%2], path="../plots/visualisation/after/")
            construct_seperation_plot(df, df.columns[:-1], fm, std_dev=0.5, show=0, save=1)

            # visualising clusters
            print("\nSaving cluster plot")
            fm = FigureMate(heading=title_prefix[i%2] + "Cluster visualisation post feature engineering", legend_labels=axes_labels[i%2], prefix=0, path="../plots/visualisation/after/")
            construct_cluster_plot(df, df.columns[:-1], fm, dimensions=3, show=0, save=1)

        # saving final datasets
        print("\n*** Saving %s ***" % fnames[i])
        save_dataset(df,
            fnames[i] + "_unsupervised",
            "../datasets/transformed/postUnsupervised/",
            save_as="obj")

if __name__ == "__main__":
    main()
