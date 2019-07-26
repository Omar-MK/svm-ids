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
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

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


def get_FAMD_components(df, n_components):
    """
    This method returns the principal components of a dataframe containing both categorical and numerical features as well as a column class.
    The categorical features must be formatted with string format and should not be dummy or one-hot encoded.
    Note: the whole dataframe should be passed including the class column - this column should be the last column in the dataframe.
    """
    famd = prince.FAMD(n_components=n_components,
        n_iter=3,
        copy=True,
        check_input=False,
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


def drop_outliers(df, cols, threshold, verbose=False):
    """
    This is an optimised alternative to get_outliers. The method will find outliers for each class in the dataframe (where the class or labels column is the last column) and drop them returning a dataframe with removed outliers.
    """
    # initialise empty dataframe object (new_df)
    new_df = None
    # iterate over each label in df
    j = 0
    for label in set(df.iloc[:, -1]):
        # take the sub-df of the current label
        sub_df = df[df.iloc[:, -1] == label]
        prior_sample_count = len(sub_df)
        # initialise list to hold index values of rows to be dropped
        rows_to_drop = []
        # iterate over each column
        for col in cols:
            # find the maximum distance from mean allowable in the column
            mean = sub_df.loc[:, col].mean()
            max_dist_from_mean = sub_df.loc[:, col].std() * threshold
            # iterate over the rows of the column
            i = 0
            for val in sub_df.loc[:, col].values:
                # if a row value violates condition add true to list
                if abs(mean - val) > max_dist_from_mean:
                    rows_to_drop += [i]
                i += 1
        # drop rows with outliers
        index_vals = []
        sub_df = sub_df.drop(sub_df.index[rows_to_drop])
        if verbose:
            print("sample count for " + str(label) + " prior to outlier removal: ", prior_sample_count)
            print("sample count after removal: ", len(sub_df))
        # concatenate data to new_df
        if j == 0:
            new_df = sub_df
        else:
            new_df = new_df.append(sub_df, ignore_index=True)
        j += 1

    return new_df


def choose_PCA_components(df, categotical_cols='', save=False, path=''):
    """
    This method searches for an optimal yet reduced number of principal
    components of a dataframe. The categorical column names should be passed to
    prevent PCA from being carried out on them. By default this method will
    plot the results of the search for principal components.
    Note: the whole dataframe should be passed including the class column - this column should be the last column in the dataframe.
    This method is adapted from: https://scikit-learn.org/stable/auto_examples/compose/plot_digits_pipe.html#sphx-glr-auto-examples-compose-plot-digits-pipe-py
    """
    df_cat = df[categotical_cols]
    num_cols = [x for x in df.columns if x not in categotical_cols]
    df_num = df[num_cols]
    svc = SGDClassifier(loss="hinge", penalty="l2", max_iter=10000, tol=1e-5, n_jobs=-1, learning_rate="adaptive", early_stopping=True, class_weight="balanced", eta0=1)
    pca = PCA()
    pipe = Pipeline(steps=[('pca', pca), ('svc', svc)])
    param_grid = {
        "pca__n_components": get_component_numbers(len(df_num.columns), 20),
        "svc__alpha": np.logspace(-2, 2, 10)
    }
    search = GridSearchCV(pipe, param_grid, iid=False, cv=5)
    search.fit(df_num.iloc[:, :-1], df_num.iloc[:, -1])
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)

    # Plotting PCA spectrum
    pca.fit(df_num.iloc[:, :-1])
    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
    ax0.plot(pca.explained_variance_ratio_, linewidth=2)
    ax0.set_ylabel("PCA explained variance")
    ax0.axvline(search.best_estimator_.named_steps["pca"].n_components, linestyle=':', label="n_components chosen")
    ax0.legend(prop=dict(size=12))
    if save:
        plt.savefig(path + "explained_variance")
    # For each number of components, find the best classifier results
    results = pd.DataFrame(search.cv_results_)
    components_col = "param_pca__n_components"
    best_clfs = results.groupby(components_col).apply(
        lambda g: g.nlargest(1, "mean_test_score"))
    best_clfs.plot(x=components_col, y="mean_test_score", yerr='std_test_score',
                   legend=False, ax=ax1)
    ax1.set_ylabel("Classification accuracy (val)")
    ax1.set_xlabel("n_components")
    plt.tight_layout()
    if save:
        plt.savefig(path + "classification_accuracy")
    plt.show()
    n_components = input("Enter number of components chosen: ")
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(df_num.iloc[:, :-1])
    pc_df = pd.DataFrame(data=pcs)
    # re-constructing dataframe and returning
    return pd.concat([pc_df, df_num.iloc[:, -1]], axis=1)

def get_component_numbers(num, splits):
    """
    This method attempts to split a number into equal parts and returns a list
    of the locations of these parts. In cases where the number cannot be split
    into n parts, the method will iteratively attempt to split the number into
    n-1 parts.
    """
    splits = None
    for i in range(20, 1, -1):
        splits = split_into_n_parts(num - 1, i)
        if splits is not None:
            break
    if splits is None:
        return np.arange(num)[1:]
    component_nums = []
    prev = 0
    for num in splits:
        component_nums += [prev + num]
        prev += num
    return component_nums


def split_into_n_parts(x, n):
    """
    This method splits a number into n parts such that the difference between
    the lagest and smallest part are minimised.
    Adapted from: https://www.geeksforgeeks.org/split-the-number-into-n-parts-such-that-difference-between-the-smallest-and-the-largest-part-is-minimum/
    """
    result = []
    if(x < n):
        return None
    elif (x % n == 0):
        for i in range(n):
            result += [x//n]
    else:
        zp = n - (x % n)
        pp = x//n
        for i in range(n):
            if(i>= zp):
                result += [pp + 1]
            else:
                result += [pp]
    return result
