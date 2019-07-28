"""
Author: Omar M Khalil - 170018405

This is the third part of the exploration process.
This script clusters the datasets created in ExplorationAndEncoding_pt2 such
that the sample counts for each class are balanced.

In this script a loop is entered iterating over the training and testing sets where:
1. duplicate samples are dropped
2. the data in all features is scaled from -1 to 1
3. Clustering is carried out on the training sets datasets (num cols only).
4. Plots to visualise training data are created
5. The generated datasets are saved
"""

import pickle
import pandas as pd
from Plotting import *
from DataTransformation import *
from CleaningAndAugmentation import save_dataset
from FigureMate import FigureMate
from sklearn import preprocessing
from sklearn.decomposition import PCA


def main():
    fnames = ["trainingset_augmented_multiclass",
              "trainingset_augmented_binary",
              "testingset_augmented_multiclass",
              "testingset_augmented_binary"]

    title_prefix = ["Multiclass Training Set ",
                    "Binary Training Set ",
                    "Multiclass Testing Set ",
                    "Binary Testing Set "]

    # loading label encodings
    mcl = pickle.load(open("../datasets/transformed/multiclass_label_encodings.obj", "rb"))
    bl = pickle.load(open("../datasets/transformed/binary_label_encodings.obj", "rb"))
    axes_labels = [mcl, bl]

    # loading datasets
    datasets = pickle.load(open("../datasets/transformed/datasets_end_pt2.obj", "rb"))

    # initialising a min max scaler
    scaler = preprocessing.MinMaxScaler().fit(datasets[0].iloc[:, :-27])
    # defining dataframe object label suffixes
    trans_lbls = ["", "Clustered"]
    for i in range(len(datasets)):
        print("\n*** Carrying out unsupervised methods on %s ***" % fnames[i])
        df = datasets[i]

        # dropping duplicates
        df = df.drop_duplicates()

        # scaling numerical data
        scaled_data = scaler.transform(df.iloc[:, :-27].values)
        X = pd.DataFrame(scaled_data, index=df.index, columns=df.columns[:-27])
        # re-joining categorical + numerical X, and y
        df = pd.concat([X, df.iloc[:, -27:]], axis=1)
        # if current dataframes are training sets
        if i < 2:
            # getting df without categorical cols
            df_num = pd.concat([df.iloc[:, :-27], df.iloc[:, -1]])
            # balancing sample counts for each class through clustering
            print("*** Constructing Clustered PCA dataset ***")
            df_clustered = balance_sample_counts(df_num,
                                                 max_clusters=3000,
                                                 mini_batch_multiplier=3,
                                                 verbose=True)
            for (df, lbl) in zip([df, df_clustered], trans_lbls):
                # plotting barchart to vis ditribution of labels
                print("Saving label count plot")
                fm = FigureMate(heading=title_prefix[i] + "Post " + lbl,
                                tick_labels=axes_labels[i%2],
                                path="../plots/visualisation/after/")
                construct_frequency_plot(df, df.columns[-1], fm, show=0, save=1)

                # re-visualising seperabilitiy of left over engineered features
                print("\nSaving seprability plot")
                fm = FigureMate(heading=title_prefix[i%2] + "Post " + lbl,
                                legend_labels=axes_labels[i%2],
                                path="../plots/visualisation/after/")
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
                                path="../plots/visualisation/after/")
                construct_cluster_plot(df,
                                       df.columns[:-1],
                                       fm,
                                       dimensions=3,
                                       show=0,
                                       save=1)
        else:
            df_clustered = df
        for (df, lbl) in zip([df, df_clustered], trans_lbls):
            # saving final datasets
            print("\n*** Saving %s ***" % fnames[i])
            save_dataset(df,
                         fnames[i] + "_" + lbl,
                         "../datasets/transformed/postUnsupervised/",
                         save_as="obj")

if __name__ == "__main__":
    main()
