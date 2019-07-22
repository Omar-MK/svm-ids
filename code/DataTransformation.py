"""
Author: Omar M Khalil - 170018405

This file contains methods used for tranforming the raw data

"""
import prince
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from scipy.stats.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def split_train_test(X, y, propOfTestSet, seed=99):
    """
    This method is used to split the data into training, and testing sets
    """
    # The number of Test set row numbers are calculated to allow us to split the data into training and testing sets.
    if isinstance(X, pd.DataFrame):
        lenOfTestSet = int(len(X.iloc[:,0]) * propOfTestSet)
    else:
        lenOfTestSet = int(len(X) * propOfTestSet)

    # Using Bernoulli sampling built into sklearn.model_selection, we can sample cases in the original dataset into the subsets

    # splitting data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=lenOfTestSet, random_state=seed)

    return x_train, x_test, y_train, y_test


def balance_sample_counts(df, max_clusters='', mini_batch_multiplier=3, verbose=False):
    """
    This method balances the sample counts for each class in a pandas dataframe, where the class column is the last column. The method uses KMeans clustering to do this, which means the data should not contain categorical features. If the dataframe contains categorical features, consider using FAMD to derive numerical features.
    """
    # checking all columns contain numerical data only
    for col in df.columns:
        if str(df[col].dtype) != "float64" and str(df[col].dtype) != "int64":
            if verbose:
                print("Found column of type %s in dataframe, attempting to cast to float..." % df[col].dtype)
            try:
                df[col] = df.loc[:, col].astype('float64')
                if verbose:
                    print("%s column changed to float64 format" % col)
            except Exception as e:
                if verbose:
                    print(e)
    # obtaining number of rows of smallest class
    rows_smallest_class = len(df.index)
    for label in set(df.iloc[:, -1]):
        rows_current_class = len(df[df.iloc[:, -1] == label].index)
        if rows_current_class < rows_smallest_class:
            rows_smallest_class = rows_current_class
    # setting max clusters
    if max_clusters == '' or max_clusters > rows_smallest_class:
        max_clusters = rows_smallest_class

    # carrying out clustering
    data = None
    i = 0
    for label in set(df.iloc[:, -1]):
        n_samples = len(df[df.iloc[:, -1] == label].index)
        # creating kmeans model
        batch_size = max_clusters * mini_batch_multiplier
        if verbose:
            print("Label ", label, " has ", n_samples, "samples,", "clustering sample to ",  max_clusters, " samples in batches of ", batch_size)
        kmeans = MiniBatchKMeans(n_clusters=max_clusters, batch_size=batch_size, compute_labels=False)
        # fitting the model to samples
        X = df[df.iloc[:, -1] == label].iloc[:, :-1]
        kmeans.fit(X)
        temp_data = np.concatenate((kmeans.cluster_centers_,  [[label]] * len(kmeans.cluster_centers_)), axis=1)
        if i == 0:
            data = temp_data
        else:
            data = np.concatenate((data, temp_data), axis=0)
        i += 1

    return pd.DataFrame(data=data)


def get_principal_components(df, n_components):
    """
    This method returns the principal components of a dataframe containing both categorical and numerical features as well as a column class.
    The categorical features must be formatted with string format and should not be dummy or one-hot encoded.
    Note: the whole dataframe should be passed including the class column - this column should be the last column in the dataframe.
    """
    famd = prince.FAMD(n_components=n_components,
        n_iter=3,
        copy=True,
        check_input=True,
        engine='auto')
    # fitting famd to data to calculate reduced dimensions
    famd = famd.fit(df.iloc[:, :-1])
    # replacing df with df containing data of projected dimensions and labels
    df = pd.concat([famd.row_coordinates(df.iloc[:, :-1]), df.iloc[:, -1]], axis=1)
    return df


def get_correlated_features(df, cols=None, classification=True, threshold=0.9, verbose=False):
    """
    finds covariate features in a dataframe. If a list of column names are passed, the method will only check for covariance within those columns. The list of columns passed should only consist of numerical features.
    The method expects the y labels to be the last column in the dataframe for classification task datasets.
    Note: by default this method treats the dataset passed in as a classification dataset.
    """
    if cols is None:
        cols = df.columns

    common_correlated_features = []
    if not classification:
        common_correlated_features = find_correlations(df, cols, threshold)
    else:
        # getting list of correlated features
        correlated_features = []
        # checking for co-variance of features for each class independently
        for label in set(df.iloc[:, -1]):
            correlated_features += [find_correlations(df[df.iloc[:, -1] == label], cols, threshold)]

        # now finding correlated features common in all classes.
        # common correlated features will be a subset of the smallest list of
        # correlations.
        # finding the length of the smallest list of correlations within
        # correlated_features
        numOfCorrelations  = len(correlated_features[0])
        smallest_list_index = 0
        for i in range(1, len(correlated_features)):
            if len(correlated_features[i]) < numOfCorrelations:
                numOfCorrelations = len(correlated_features[i])
                smallest_list_index = i

        # now iterating over the correlations in the smaller list and checking if the same correlations exist in the other lists
        for i in range(0, numOfCorrelations):
            correlations_found = 0 # to count the number of times the same correlation exists in all other correlations lists
            # loop over all correlation lists before the smallest list
            for j in range(0, smallest_list_index):
                if correlated_features[smallest_list_index][i] in correlated_features[j]:
                    correlations_found += 1
            # loop over all correlation lists after the smallest list
            for j in range(smallest_list_index+1, len(correlated_features)):
                if correlated_features[smallest_list_index][i] in correlated_features[j]:
                    correlations_found += 1
            if correlations_found == len(correlated_features) - 1:
                common_correlated_features += [correlated_features[smallest_list_index][i]]

    if verbose:
        print("number of correlations found: ", len(common_correlated_features))
        print("Correlated features are :", common_correlated_features)

    return common_correlated_features


def find_correlations(df, cols, threshold):
    """
    remove_correlated_features helper method.
    Returns a list of tuples of correlated features in the passed columns of a dataframe.
    """
    correlated_features = []
    # iterating over the outer column
    for i in range(len(cols)):
        col1 = cols[i]
        # iterating over the inner column
        for col2 in cols[i+1:]:
            pearsonrValue = pearsonr(df[col1], df[col2])[0]
            if pearsonrValue > threshold or pearsonrValue < -threshold:
                correlated_features += [(col1, col2)]
    return correlated_features

def balance_by_sampling(df):
    """
    This method balances the sample count in a pandas dataframe for each class (in the last column) such that all classes are equal to the count of the smallest class.
    """
    # getting count of samples smallest class
    count_smallest = len(df[df.iloc[:, -1] == list(set(df.iloc[:, -1]))[0]])
    for label in set(df.iloc[:, -1]):
        count_current = len(df[df.iloc[:, -1] == label])
        if count_current < count_smallest:
            count_smallest = count_current

    balanced_df = None
    i = 0
    # randomly sampling count_smallest rows and placing them in balanced_df
    for label in set(df.iloc[:, -1]):
        current_df = df[df.iloc[:, -1] == label]
        sampled_rows = current_df.sample(count_smallest)
        if i == 0:
            balanced_df = sampled_rows
        else:
            balanced_df = pd.concat([balanced_df, sampled_rows])
        i += 1
    return balanced_df
