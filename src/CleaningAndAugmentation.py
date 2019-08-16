"""
Author: 170018405

This file contains code used to carry out the inital data cleaning and feature augmentation. Running this file generates a single combined and augmented dataset from the 8 UNB datasets.
"""

from DataWrangler import DataWrangler
from sklearn import preprocessing
from DataChecking import *
from DataTransformation import *
import pandas as pd
import numpy as np
import pickle


def get_av_differences(df, feature, n):
    """
    Calculates the average difference between n values in a feature.
    for example: Given the following col: A, B, C, D, E and n = 5
    results at row E is ((E - D) + (D - C) + (C - B) + (B - A)) / 3
    """
    result = [0]
    values = df.loc[:, feature].values
    for i in range(1, len(values)):
        j = i - 1
        diff = 0
        while j > -1 and (i - j) < n:
            diff += abs(float(values[j + 1] - values[j])) / 10**9
            j -= 1
        av_diff = diff / (n-1)
        result += [av_diff]
    return result


def get_diff_from_last(df, features):
    """
    Given two features (key and value) calculates the difference between the current and last values with the same key.
    If a key has never occured before, a null value is placed since a difference cannot be calculated.
    """
    if len(features) != 2:
        return None
    result = [None]
    keys = df.loc[:, features[0]].values
    values = df.loc[:, features[1]].values
    for i in range(1, len(keys)):
        value = values[i]
        diff = None
        for j in range(i-1, -1, -1):
            if keys[j] == keys[i]:
                diff = abs(float(values[i] - values[j])) / 10**9
                break
        result += [diff]
    return result


def get_n_prior_occurances(df, columns, n_list, frac=False):
    """
    counts the number of times the values at row i in each of the columns in columns occur together in the prior n rows.
    Returns a list of lists of the counts for each n value given in n_list.
    Params:
    df - pandas dataframe containing the data
    columns - list of column names in the dataframe
    n_lisr - a list of the number of prior cells to count prior occurances in
    frac - (boolean) converts counts in output list to percentages if true
    """

    data_lists = []
    for col in columns:
        data_lists += [df.loc[:, col].values]
    zipped_list = list(zip(*data_lists))
    results = []
    for n in n_list:
        occurance_counts = []
        for i in range(len(zipped_list)):
            current_val = zipped_list[i]
            prior_n_count = 1
            vals = zipped_list[max(0, i - (n - 1)):i + 1]
            if frac:
                occurance_counts += [float(vals.count(current_val)/n)]
            else:
                occurance_counts += [vals.count(current_val)]
        results += [occurance_counts]
    return results


def fill_missing(list, fill_value=0):
    """
    Fills None elements in a list with fill_value. By default fill_value = 0.
    """
    adjusted_list = []
    for val in list:
        if np.isnan(val):
            adjusted_list += [fill_value]
        else:
            adjusted_list += [val]
    return adjusted_list


def change_everything_except(list, except_val, to_val):
    """
    Changes all values in a list which are not a given value to a given value.
    """
    adjusted_list = []
    for val in list:
        if val != except_val:
            adjusted_list += [to_val]
        else:
            adjusted_list += [except_val]
    return adjusted_list


def save_dataset(df, fname, path, save_as="csv"):
    fname = fname.replace(".csv", "")
    fname = fname.replace(".obj", "")
    fname = fname.replace(' ', '_')
    if save_as== "csv":
        df.to_csv(path + fname + ".csv", index=False)
    elif save_as== "obj":
        pickle.dump(df, open(path + fname + ".obj", "wb"))


def main():
    datasets = ["Monday-WorkingHours.pcap_ISCX.csv",
                "Tuesday-WorkingHours.pcap_ISCX.csv",
                "Wednesday-workingHours.pcap_ISCX.csv",
                "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
                "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
                "Friday-WorkingHours-Morning.pcap_ISCX.csv",
                "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
                "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"]

    # expected number of classes in each dataset
    expected_classes = [1, 3, 6, 4, 2, 2, 2, 2]
    # augmented features n vals
    n_vals = [5, 10, 15, 20]

    # Beginning cleaning and feature augmentation process
    sameCols = False
    db = DataWrangler()
    i = 0
    for dataset in datasets:
        print("Loading ", dataset)
        # loading the data set
        db_current = DataWrangler("../datasets/CIC_IDS_17/TrafficLabelling/" + dataset, encoding = "cp1252", skipspace=True)
        # dropping rows with null values
        print("Checking for null values")
        db_current.drop_null_values(verbose=True)
        # checking if the data types of the columns match the data types of the majority of contained data, if there are "mixed columns", an auto fix is attempted.
        print("Checking data types...")
        if len(db_current.check_data_types(verbose=True)) > 0:
            db_current.autofix_mixed_columns(convert_datetime=True, verbose=True)

        # grabbing the cleaned dataset from the DataWrangler object
        df_current = db_current.get_df()
        cols = df_current.columns

        # Adding features to keep track of time difference between connections
        print("Augmenting temporal features...")
        f_name = "Av time diff between last n connections, n="
        for n in n_vals:
            av_times = get_av_differences(df_current, cols[6], n)
            df_current.insert(len(cols) - 1, f_name + str(n), av_times)

        # Adding a feature to represent the time difference since the last connection from the same IP.
        # Note due to the nature of this feature, new connections from new IP addresses cannot have a time since last connection. These values cannot be replaced with a 0 since this implies that it has been 0 seconds since this IP address last connected. Similarly, it cannot be left blank. One of the ways to represent these values is to employ a concept from laplace smoothing - That is, an item which is unseen, can essentially be represented as having the probability to occur as that of a rare item (that which has been observed once) occurs. Thus, we can give rare cases times differences of the longest difference in the training sets. Shortly after splitting the data into training and testing sets, the max value in the training set is obtained, and any empty cells are replaced with this value (this is carried out at later in the code).
        diff_from_last_vals = get_diff_from_last(df_current, [cols[1], cols[6]])
        df_current.insert(len(cols) - 1, "time since last conn", diff_from_last_vals)

        # Adding features used to keep track of connections
        print("Augmenting features to keep track of connections by IP addresses...")
        features = [[cols[1]], [cols[3]], [cols[1], cols[3]], [cols[1], cols[4]]]
        feature_names = ["% connections s-IP within n=", "% connections d-IP within n=", "% connections s-IP to d-IP within n=", "% connections s-IP to d-port within n="]
        for f_name, feature in list(zip(feature_names, features)):
            prior_occurances = get_n_prior_occurances(df_current, feature, n_vals, frac=True)
            j = 0
            for n in n_vals:
                df_current.insert(len(cols) - 1, f_name + str(n), prior_occurances[j])
                j += 1

        # The nature of the augmented features above mean that the first n_val[-1] rows do not contain correct readings. Thus, those rows are deleted.
        df_current = df_current.iloc[n_vals[-1]:]

        # checking and printing unique attacks in current dataset
        unique_attacks = list(set(df_current.Label))
        print("Unique attacks in", dataset, ": ", unique_attacks)

        print("Adding datapoints from current file to main database...")
        if len(unique_attacks) == expected_classes[i]:
            print("Number of unique classes as expected")
            db_current.set_df(df_current)
            # appending fixed data set (db_current) to db
            print("cocatenation sucess: ", db.concat(db_current, sameCols=sameCols, warning=True))
            sameCols = True
            i += 1
        else:
            raise Exception("Unexpected number of unique classes")


    # Getting the cleaned complete dataset
    df = db.get_df()

    # the Flow ID, Source and Destination IP, timestap, and source port features are dropped
    print("Dropping no longer needed features...")
    cols = df.columns
    df = df.drop(columns=[cols[0], cols[1], cols[2], cols[3], cols[6]])

    # encoding class labels
    print("Encoding class labels...")
    le = preprocessing.LabelEncoder()
    le.fit(df.Label)
    df.Label = le.transform(df.Label)

    # writing multi-class names and encoding to csv
    print("Creating csv with class keys and values...")
    mapping_names = dict(zip(le.classes_, le.transform(le.classes_)))
    pickle.dump(mapping_names, open("../datasets/cleaned/multiclass_encoding.obj", "wb"))

    # splitting data into 80:20 training and testing sets
    print("Splitting data into training and testing sets...")
    X = df.iloc[:,:-1]
    y = df.Label
    X_train, X_test, y_train, y_test = split_train_test(X, y, 0.2, seed=99)

    # Using the training set, the max difference is found
    max_diff = np.nanmax(np.array(X_train["time since last conn"], dtype=np.float64))

    # Now filling in empty cells for time diff column (discussed above)
    print("Empty time difference values are filled with max time diff from training data")
    df.loc[:,"time since last conn"] = fill_missing(df["time since last conn"], max_diff)

    # removing duplicated rows
    print("Calculating number of duplicated rows...")
    print("Number of duplicated rows: ", df.duplicated().sum())
    df = df.drop_duplicates(keep='first')


    # Resplitting filled df
    X = df.iloc[:,:-1]
    y = df.Label
    X_train, X_test, y_train, y_test = split_train_test(X, y, 0.2, seed=99)


    # X_train - y_train, and x_test - y_test are concatenated into seperate dataframes

    # creating multiclass dataframes
    print("Creating multiclass dataframes...")
    training_data = pd.concat([X_train, y_train], axis=1)
    testing_data = pd.concat([X_test, y_test], axis=1)

    # creating binary dataframes
    print("Creating binary dataframes...")
    training_data_binary = training_data.copy()
    testing_data_binary = testing_data.copy()
    if mapping_names["BENIGN"] == 0:
        training_data_binary.loc[training_data_binary.Label != mapping_names["BENIGN"], 'Label'] = 1
        testing_data_binary.loc[testing_data_binary.Label != mapping_names["BENIGN"], 'Label'] = 1
        print("Attack = 1 in binary data")
    else:
        training_data_binary.loc[training_data_binary.Label != mapping_names["BENIGN"], 'Label'] = 0
        testing_data_binary.loc[testing_data_binary.Label != mapping_names["BENIGN"], 'Label'] = 0
        print("Attack = 0 in binary data")


    print("Saving datasets...")
    # saving augmented binary and multiclass training and test sets
    path = "../datasets/cleaned/"
    training_data = training_data.sort_values(by=cols[-1])
    save_dataset(training_data, "trainingset_augmented_multiclass", path, save_as="obj")
    testing_data = testing_data.sort_values(by=cols[-1])
    save_dataset(testing_data, "testingset_augmented_multiclass", path, save_as="obj")
    training_data_binary = training_data_binary.sort_values(by=cols[-1])
    save_dataset(training_data_binary, "trainingset_augmented_binary", path, save_as="obj")
    testing_data_binary = testing_data_binary.sort_values(by=cols[-1])
    save_dataset(testing_data_binary, "testingset_augmented_binary", path, save_as="obj")


if __name__ == "__main__":
    main()
