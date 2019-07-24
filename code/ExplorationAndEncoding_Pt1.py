"""
Author: Omar M Khalil - 170018405

This is the first part of the exploration and encoding process. The program will prepare and save the final datasets for the pre-supervised task, and then save an object  (containing datasets) ready for additional work for the post-supervised task.

The file contains code used to:
1. Load and confirm datasets have no unexpected data types.
2. Check balance of data points per class and save visualisations of these findings.
3. Remove low count classes (Heartbleed, Infiltration), and combine other similar clases (Web attacks) together to improve balance of classes.
4. Check counts of unique values in categorical features. This is important to since it can indicate if a categorical feature can be dropped, and if not, which values to represent in using dummy encoding.
5. Plot results from step 4.
6. Plot categorical features vs class labels to get an idea of how well each categorical feature influences the output - this is done for the same reason as step 4.
7. Reduce categories in the cateogrical features such that the rest of the cateogires can be dummy encoded while not adding too many features.
8. Print descriptive stats, and outlier presence info in numerical features.
9. Normalise/scale numerical features.
10. Plot class sperability plots, to understand if the data is linearly seperable.
11. Perform FAMD feature projection to visualise the clusters of the datasets in 3D.
12. Dummy encode the categorical features for the pre-supervised datasets.
13. Save the pre-supervised datasets.
14. Save objects containing datasets to be used for post-supervised task.
"""

import pandas as pd
import pickle
from DataChecking import *
from DataTransformation import *
from Plotting import *
from sklearn.preprocessing import StandardScaler
from FigureMate import FigureMate

def save_datasets(datasets, fnames, path, save_as="csv"):
    for (df, fname) in zip(datasets, fnames):
        fname = fname.replace(".csv", "")
        fname = fname.replace(".obj", "")
        # creating non-augmented dataframe
        to_drop = df.columns[-27:-1]
        df_reduced = df.drop(columns=to_drop)
        if save_as == "csv":
            # saving augmented dataset
            df.to_csv(path + fname + ".csv", index=False)
            # saving non-augmented dataset
            df_reduced.to_csv((path + fname + ".csv")
                .replace('_augmented', ''), index=False)
        elif save_as == "obj":
            # saving augmented dataset
            pickle.dump(df, open(path + fname + ".obj", "wb"))
            # saving non-augmented dataset
            pickle.dump(df_reduced,
                open((path + fname + ".obj").replace('_augmented', ''), "wb"))


def plot_label_counts(datasets, title_prefix, axes_labels, path_suffix=''):
    for i in range(len(datasets)):
        # plotting barchart to vis ditribution of labels
        fm = FigureMate(heading=title_prefix[i], tick_labels=axes_labels[i%2], path="../plots/visualisation/before/" + path_suffix)
        construct_frequency_plot(datasets[i], datasets[i].columns[-1], fm, show=0, save=1)


def main():
    # loading the input and output data
    print("*** Loading datasets ***")
    path = "../datasets/cleaned/"
    fnames = ["trainingset_augmented_multiclass",
        "trainingset_augmented_binary",
        "testingset_augmented_multiclass",
        "testingset_augmented_binary"]

    datasets = []
    for file in fnames:
        datasets += [pickle.load(open(path + file + ".obj", "rb"))]

    # loading class encodings dictionary
    class_encodings = pickle.load(open(path + "multiclass_encoding.obj", "rb"))

    # checking for unexpected data formats during loading
    print("\n*** Checking for unexpected data formats ***")
    # extensive checking of datatypes and data cleaning was already conducted in
    # cleaningAndAugmentation.py. The following check of column datatype is a
    # quick method to check loading of datasets was carried out successfully.
    for df in datasets:
        if 'object' in df.columns:
            raise ValueError("Unexpected data type in dataset")
        else:
            print("All data types are as expected")

    # checking class counts
    print("\n*** Checking no. of samples in the dataframes for each class ***")
    for df, fname in zip(datasets[:2], fnames[:2]):
        print(fname + "\n", df.Label.value_counts())

    # saving visualisations of class counts
    # collecting class labels for multiclass axes labels
    mcl = list(class_encodings.keys())
    # defining title labels
    title_prefix = ['Multiclass Training Set ',
        'Binary Training Set ',
        'Multiclass Testing Set ',
        'Binary Testing Set ']
    axes_labels = [mcl, ["BENIGN", "Attack"]]

    plot_label_counts(datasets[:2], title_prefix[:2], axes_labels)

    # the results above indicate that clustering is required to balance out the
    # classes for both binary and multiclass training sets. Moreover, Heartbleed
    # and Infiltration attacks need to be dropped and all web attacks should be
    # combined.

    # dropping Heartbleed and Infiltration attacks, then joining web attacks
    # note this will only affect multiclass datasets
    print("*** Dropping low count classes and reshuffling labels ***")
    for i in range(len(datasets)):
        df = datasets[i]
        df = df[df.Label != 8] # Heartbleed removed
        df = df[df.Label != 9] # Infiltration removed
        df.Label = df.Label.replace([10], 8) # moving PortScan to 8
        df.Label = df.Label.replace([11], 9) # moving SSH_Patator to 9
        df.Label = df.Label.replace([12, 13, 14], 10) # combining web attacks
        datasets[i] = df

    # updating mcl
    del mcl[8], mcl[8], mcl[11], mcl[11]
    mcl[10] = "Web Attack"
    print(mcl)
    # saving mcl and binary class encodings
    pickle.dump(mcl,
        open("../datasets/transformed/multiclass_label_encodings.obj", "wb"))
    pickle.dump(["BENIGN", "Attack"],
        open("../datasets/transformed/binary_label_encodings.obj", "wb"))

    plot_label_counts(datasets[:2], title_prefix, axes_labels, path_suffix="changed_")

    # checking counts of values of categorical attributes. This is important to
    # see if the categorical attributes can be dropped. If not, the
    # K-Prototypes algorithm could be used instead of k-means to cluster the
    # data to reduce instances.
    cat_cols = datasets[0].columns[0:2]
    print("\n*** Checking the counts of categorical data in training sets ***")
    # plotting counts of categorical attributes
    cat_df = datasets[0]
    for col in cat_cols:
        num_cats = len(set(datasets[0][col]))
        print("number of unique " + col + "s: ", num_cats)
        print_unique_counts(datasets[0], [col])
        if num_cats > 25:
            cat_df = datasets[0].loc[datasets[0][col].isin(datasets[0][col].value_counts().index[:25])]

        fm = FigureMate(heading="Training Set",
            path="../plots/visualisation/before/")
        construct_frequency_plot(cat_df, col, fm, show=0, save=1)

    # plotting categorical features vs labels to get a better understanding of
    # correlations
    # the following filters applied to the main dataset ensure that only port
    # numbers and protocols that occur with a class which is not Benign are
    # plotted.
    cat_df = datasets[0][datasets[0].Label != 0]
    cat_df = cat_df.loc[cat_df[cat_cols[0]]
        .isin(cat_df[cat_cols[0]]
        .value_counts()
        .index[:25])]

    i = 0
    for dataset in datasets[:2]:
        fm = FigureMate(heading=title_prefix[i] + " (attacks only)",
            path="../plots/visualisation/before/" + title_prefix[i] + 'attacks_only_')
        construct_frequency_plot(cat_df, cat_cols[0], fm, show=0, save=1)
        construct_box_plot(cat_df, cat_cols, ["Label"], fm, show=0, save=1)
        construct_violin_plot(cat_df, cat_cols, ["Label"], fm, show=0, save=1)
        i += 1

    # the above plots show that protocol 6 is most important when deciding type
    # of attack. Similarly, the majority of destination port numbers either
    # indicate Benign or FTP patator attack. The majority of connections are to
    # ports 53, 80, and 443. However, when only the destination ports leading
    # to an attack class are plotted, the top attacks are: 80, 21, 22, and
    # 8080. A decision is made to:
    # 1. Keep only port numbers: 80, 21, 22, 8080.. everything else will be 0
    # 2. Keep only protocol 6, the rest are converted to 0.
    print("\n*** Removing categories within categorical features not required ***")
    for i in range(len(datasets)):
        df = datasets[i]
        dports_replace = set(df.iloc[:, 0])
        dports_replace -= {80, 21, 22, 8080}
        df.iloc[:, 0] = df.iloc[:, 0].replace(list(dports_replace), 0)
        prot_replace = set(df.iloc[:, 1])
        prot_replace -= {6}
        df.iloc[:, 1] = df.iloc[:, 1].replace(list(prot_replace), 0)
        # converting categorical cols to string format for FAMD pca
        df.iloc[:, 0] = df.iloc[:, 0].astype(str)
        df.iloc[:, 1] = df.iloc[:, 1].astype(str)
        datasets[i] = df

    # now exploring the numerical features

    # given the difficulty of visualising all 98 numerical features
    # independently, we can perform some statistics to help us understand the
    # distribution of data in each feature as well as how well the features
    # help seperation in the classification task. First we can print
    # descriptive statists. Then, assuming a normal distribution, we can
    # estimate the number of outliers present. I.e. If one of the values in a
    # feature is very far from the others of the same class, this will indicate
    # the spread of the data for each label. To check for this, the distance
    # each value is from the population mean in standard deviations is
    # calculated. Values further than 3 SD from the mean are deemed as outliers.
    print("\n*** Printing discriptive stats and checking for outliers in the data for each class independently using a threashold of 3 * SD ***")
    for df in datasets[:2]:
        for label in set(df.iloc[:, -1]):
            print("\nShowing discriptive stats for class: ", label)
            # print(df[df.Label == label].describe())
            print("\nOutlier information")
            # print_outlier_information(df[df.Label == label].iloc[:, 2:-1], 3)

    # this first round result shows that around a 5th of all "outliers" for
    # every class come from a single column, yet this column is different for
    # different classes. Moreover, some classes have as many as 33.4% of all
    # rows containing outliers. These rows could be removed at this point.
    # However, since clustering will be used to reduce the data, this is not
    # carried out.

    # normalising numerical data to have a mean ~0 and variance of 1, aids in
    # clustering process and visualisations as well as SVM model building. Also
    # dummy encoding categorical data.
    print("\n*** Normalising the numerical features and dummy encoding categorical features ***")
    scaler = StandardScaler().fit(datasets[0].iloc[:, 2:-1])
    # using the above scaler fitted on the training data, scailing all numerical
    # training and testing dataset values
    pre_unsupervised_dfs = []
    for i in range(len(datasets)):
        df = datasets[i]
        df.iloc[:, 2:-1] = scaler.transform(df.iloc[:, 2:-1])
        datasets[i] = df
        y = df.iloc[:,-1]
        X = pd.get_dummies(df.iloc[:,:-1],
            columns=df.columns[:2],
            drop_first=True)
        df = pd.concat([X, y], axis=1)
        pre_unsupervised_dfs += [df]

        # the following plot allows us to see if the numerical features for both
        # classes are seperable. The more seperable they are the better the
        # classifier will be.
        if i < 2:
            fm = FigureMate(heading=title_prefix[i%2] + "Cluster visualisation pre feature engineering", prefix=0, path="../plots/visualisation/before/")
            construct_cluster_plot(df, df.columns[:-27], fm, dimensions=3, show=0, save=1)
            fm = FigureMate(heading=title_prefix[i%2], path="../plots/visualisation/before/")
            construct_seperation_plot(df, df.columns[:-27], fm, std_dev=0.5, show=0, save=1)


    # saving the cleaned and encoded vanilla datasets
    print("\n*** Saving encoded pre-unsupervised datasets ***")
    save_datasets(pre_unsupervised_dfs, fnames, "../datasets/transformed/preUnsupervised/", save_as="obj")

    # saving datasets for second part of ExplorationAndEncoding
    pickle.dump(datasets, open("../datasets/transformed/datasets_end_pt1.obj", "wb"))


if __name__ == "__main__":
    main()
