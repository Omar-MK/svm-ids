"""
Author: Omar M Khalil - 170018405

This is the second part of ExplorationAndEncoding_pt1.py and includes further changes to post-supervised datasets.

The file contains code used to:
1. Load the object containing the datasets.
2. Check for co-variate numerical features and reduce the features to remove co-variance.
3. carry out FAMD projection of the data to project the features into a lower feature space. This has the effect of turning all features into numeric features.
4. Using the numeric feautres, KMeans clustering is applied balance data point count per class.
5. Re-check for outliers, and based on user input, the outliers are removed or kept.
6. Plot the seprability plot and 3D cluster plots from steps 10 and 11 are remade to visualise effect of changes.
7. Save the post-superivsed datasets.
"""

import pandas as pd
import pickle
from DataChecking import *
from DataTransformation import *
from Plotting import *
from cleaningAndAugmentation import save_dataset
from sklearn.preprocessing import StandardScaler
from DataWrangler import DataWrangler


def main():
    fnames = ["trainingset_augmented_multiclass",
            "trainingset_augmented_binary",
            "testingset_augmented_multiclass",
            "testingset_augmented_binary"]

    # loading stored object containing datasets
    print("*** Loading datasets.obj object ***")
    datasets = pickle.load(open("../datasets/transformed/datasets_end_pt1.obj", "rb"))

    # now applying un-supervised data quality boosting methods

    # first We can check to see if any numerical features are highly correlated
    # (co-variate). This could allow us to reduce the number of features. The
    # pearson r test is used to detect if there is any correlation between
    # numerical features.
    # the following code will find the correlated features for every class and
    # append a list of tuples of the feature column numbers in
    # all_correlated_features. Note, this process is only applied to the multiclass
    # dataset since co-variate features in this dataset will also be covariate in
    # the binary class, yet not neccessarily the other way around.
    print("\n*** Checking for co-variate numerical features ***")
    common_correlated_features = get_correlated_features(datasets[0], cols=datasets[0].columns[2:-1], classification=True, threshold=0.9, verbose=True)

    # now removing correlated features
    print("removing correlated features")
    for d in range(len(datasets)):
        for [i, j] in common_correlated_features:
            if j in datasets[d]:
                datasets[d] = datasets[d].drop([j], axis=1)
    print("features remaining: ", datasets[0].columns)
    print("number of features remaining: ", len(datasets[0].columns))

    # the next step is to peform principal component analysis (FAMD) to project and
    # cat features and numerical features onto common planes. This also reduces the
    # dimensionality of the datasets. This is followed by KMeans clustering to
    # balance out the counts of rows for each class.
    print("\n*** Carrying out FAMD and clustering ***")

    for i in range(len(datasets)):
        datasets[i] = datasets[i].drop_duplicates()
        df = datasets[i]
        # getting principal components
        df = get_principal_components(df, len(datasets[0].columns) - 1)
        if i < 2:
            # balancing sample counts for each class through clustering - only
            # applied on training sets
            print("now clustering data")
            df = balance_sample_counts(df, max_clusters=3000, mini_batch_multiplier=3, verbose=True)

            # Re-checking outliers
            print("\nRe-checking for outliers after clustering")
            for label in set(df.iloc[:, -1]):
                print("\nOutlier information for class : ", label)
                print_outlier_information(df[df.iloc[:, -1] == label].iloc[:, :-1], 3)

            # re-visualising seperabilitiy of left over engineered features
            print("\nSaving seprability plots")
            plotSeperation(df, df.columns[:-1], std_dev=0.5, show=False, save=True, path='../plots/visualisation/sep_after_' + fnames[i])

            # visualising clusters
            print("\nSaving cluster plots")
            plotClusters(df, df.columns[:-1], title="Cluster visualisation post feature engineering", label_prefixes="Connection type: ", three_D=True, show=False, save=True, path='../plots/visualisation/cluster_after_' + fnames[i])

        # saving final datasets
        print("\n*** Saving %s ***" % fnames[i])
        save_dataset(df, fnames[i] + "_unsupervised", "../datasets/transformed/postUnsupervised/", save_as="obj")

if __name__ == "__main__":
    main()
