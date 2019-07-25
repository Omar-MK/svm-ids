"""
Author: Omar M Khalil - 170018405

This file contains methods used for checking raw data in a Pandas DataFrame
"""
import pandas as pd

def check_num_of_unique_values(df, expectedNumOfUniqueVals):
    """
    Checks the number of unique values in each column of a pandas dataframe match those padded in.
    """
    i = 0
    for col in df.columns:
        currentCol = set(df[col][:])
        if (len(currentCol) != expectedNumOfUniqueVals[i]):
            return False
        i += 1
    return True


def get_outliers(df, cols, threshold):
    """
    This method checks for outliers in a set of columns in a pandas dataframe.
    It will return a list of tuples (where each tuple will consist of the
    column name and the number of outliers found), and a list of index numbers
    of the rows where those outliers are found.
    """
    outliers = [] # a list of the columns that contain outliers and the count of outliers in those columns
    outlierLocations = [0] * len(df.index) # a list of the number of outliers at each row index
    for col in cols:
        rows = df.loc[:, col]
        mean = rows.mean()
        thresholdDiff = threshold * rows.std()

        # check if outliers exist in the current column
        if rows.max() > (mean + thresholdDiff) or (mean - thresholdDiff) > rows.min():
            # now we count the number of data points that are further than threashold from the mean
            count = 0
            i = 0 # index of current row in loop
            outlier_rows = [] # list of row indexes where outliers were found
            for row in rows:
                if row > (mean + thresholdDiff) or (mean - thresholdDiff) > row:
                    count += 1
                    outlier_rows += [i]
                i += 1
            outliers += [(col, count)] # updating the outliers list

            # now the outlierLocations list is updated
            for index in outlier_rows:
                outlierLocations[index] += 1

    # collecting only the rows with outliers into a single list of tuples from outlierLocations
    reduced_outlierLocations = []
    j = 0
    for i, row in df.iterrows():
        if outlierLocations[j] > 0:
            reduced_outlierLocations += [(i, outlierLocations[j])]
        j += 1
    return outliers, reduced_outlierLocations


def print_outlier_information(df, threshold):
    """
    Prints information about the distribution of outliers in a df.
    The method defines an outlier as a data point outside a threshold of
    standard deviation from the data points in the same column.
    """
    outliers, outlierLocations = get_outliers(df, df.columns[:], threshold)
    if len(outliers) > 0:
        # calculating total outliers found
        total_outliers = 0
        max_outliers_per_column = 0
        worst_col = ''
        for (col, tot) in outliers:
            total_outliers += tot
            if tot > max_outliers_per_column:
                max_outliers_per_column = tot
                worst_col = col

        print("Total outliers found: ", total_outliers)
        print("That makes up ", total_outliers / (len(df.columns) * len(df.index)) * 100, "% of the data")
        print("Maximum number of outliers in a single column: ", max_outliers_per_column)
        print("Column name: ", worst_col)
        print("The number of rows containing outliers is: ", len(outlierLocations))
        print("Percentage of rows containing outliers: ", (len(outlierLocations)/len(df.index)) * 100, "\%")


def print_unique_counts(df, cols):
    """
    Prints the counts of unique values in each column of a pandas dataframe.
    """
    for col in cols:
        print('\n' + col)
        print(df[col].value_counts())

def normalisation_required(df, cols):
    """
    Checks if normalisation is required given columns names of numerical data in a dataframe
    """
    smallest_mean = 1
    largest_mean = 0.0
    data = df[cols]
    for col in data.columns:
        if abs(data[col].mean()) > largest_mean:
            largest_mean = abs(data[col].mean())
        elif abs(data[col].mean()) < smallest_mean:
            smallest_mean = abs(data[col].mean())
    if largest_mean / smallest_mean >= 100:
        return True
    return False
