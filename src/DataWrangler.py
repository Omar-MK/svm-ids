"""
Author: Omar M Khalil - 170018405
This file contains a data wrangling class for ML tasks. The class provides methods to prepare raw data such that data set is ready for direct use in a ML task.
"""

import pandas as pd
import numpy as np
import random as rand
import re

class DataWrangler:
    """
    This is a data wrangling class for ML tasks. The class provides methods to prepare raw data such that data set is ready for direct use in a ML task.
    """

    def __init__(self, database_source='', encoding='', skipspace=False):
        if encoding == '' and database_source != '':
            self.df = pd.read_csv(database_source, skipinitialspace=skipspace)
        elif encoding != '' and database_source != '':
            self.df = pd.read_csv(database_source, encoding = encoding, skipinitialspace=skipspace)
        elif database_source == '':
            self.df = pd.DataFrame()
        # a list to store the column names containing mixed data types
        self.mixed_types = []
        # boolean used to track if check_data_types() has been used
        self.mixed_columns_checked = False
        # boolean used to track if nan values dropped
        self.nulls_dropped = True

    def set_df(self, df):
        """
        Sets a new dataframe as the current dataset.
        """
        self.df = df
        self.mixed_types = []
        self.mixed_columns_checked = False
        self.nulls_dropped = True

    def get_df(self):
        """
        returns the stored dataset as a pandas dataframe.
        """
        return self.df

    def concat(self, dw, sameCols=True, warning=True):
        """
        concatenates another db object onto current object.
        """
        # checking if either dataset has not yet dropped the null values
        if not self.nulls_dropped and dw.get_nulls_dropped():
            if self.nulls_dropped:
                # in this case self has dropped null values whereas dw has not, so dropping null values in dw
                dw.drop_null_values()
            elif dw.get_nulls_dropped():
                # in this case self has not dropped null values whereas dw has, so dropping null values in self
                self.drop_null_values()
            else:
                # both have no dropped null values so combining both.
                return self.concat_helper(dw, sameCols, warning)

        # checking if either dataset has not yet checked for mixed columns. checking for mixed columns is required since the new dataset will have the combined list of mixed columns.
        if not self.mixed_columns_checked and dw.get_mixed_columns_checked():
            if self.mixed_columns_checked:
                # in this case self has checked for mixed data types and dw not, so checking for mixed types in dw
                dw.check_data_types()
            elif dw.get_mixed_columns_checked():
                # in this case self has not checked for mixed data types and dw has, so checking for mixed types in self
                self.check_data_types()
            else:
                # have have not checked for mixed data types, so combining
                return self.concat_helper(dw, sameCols, warning)


        # now combining mixed columns and rows
        self.mixed_types = set(list(self.mixed_types) + list(dw.get_mixed_columns()))
        return self.concat_helper(dw, sameCols, warning)

    def concat_helper(self, dw, sameCols=True, warning=True):
        if sameCols:
            if list(self.df.columns) == list(dw.get_df().columns):
                self.df = pd.concat([self.df, dw.get_df()], axis=0, ignore_index=True)
                return True
            else:
                return False
        else:
            self.df = pd.concat([self.df, dw.get_df()], axis=0, ignore_index=True)
            if list(self.df.columns) != list(dw.get_df().columns) and warning:
                print("The combined datasets do not contain the same columns! NaN values added where appropriate.")
            return True

    def drop_null_values(self, verbose=False):
        """
        Drops rows containing null and inf values. If verbose mode is enbaled, the method prints stats about null values found in a dataframe.
        """
        if verbose:
            nulls = self.df.isnull()
            print("%d NaN and inf cells found in df" % nulls.sum().sum())
            columns_with_nulls = self.df.columns[nulls.any()]
            print(columns_with_nulls)
            print("Number of columns containing NaN/inf values: ", nulls.any().sum())
            print("Number of rows dropped containing NaN/inf values: ", nulls.sum(axis=1).sum())

        self.nulls_dropped = True
        self.df = self.df.replace([np.inf, -np.inf], np.nan)
        self.df = self.df.dropna()


    def check_data_types(self, verbose=False):
        """
        Checks if the data types of the columns in the object's pandas dataframe are consistent with the data types of a representative sample of the values present in the columns.
        Returns a list of column names where inconsistenices are detected.
        """
        # 40% or 20 rows (whichever is lower) of the current column are randomly sampled and stored in a list
        sampled_df = self.df.sample(n = min(int(len(self.df.index) * 0.4), 20))

        i = 0
        for col in self.df.columns:
            # now the types of the stored sampled data values are compared with the data type of the current column in the dataframe. If there is a contradiction, then the column name is stored in the mixed_types list.
            column_type = self.df.dtypes[i]
            for t in sampled_df[col]:
                if column_type != type(t) and column_type != 'datetime64[ns]':
                    # checking for digits in sampled data point and that no more than 1 alphabetical characters follow each other (except for AM/PM) within the string. If the data point matches the above conditions it means the column either contains numbers denoting time, a real scientific number, or some other numerical format. These columns are placed in the mixed_types list.
                    if re.search('\d', str(t)) != None and (re.search('[a-zA-Z]{2,}', str(t)) == None or re.search('AM|PM|am|pm', str(t)) != None):
                        self.mixed_types += [col]
                        break
            i += 1
        if verbose:
            print("The following columns were determined to contain inconsistent data types:")
            print(self.mixed_types)

        self.mixed_columns_checked = True
        self.drop_null_values()
        return self.mixed_types

    def get_mixed_columns(self):
        """
        Returns a list of columns containing mixed data types.
        """
        if not self.mixed_columns_checked:
            self.check_data_types()
        return self.mixed_types

    def autofix_mixed_columns(self, convert_datetime=False, datetime_format='%I:%M %p', verbose=False):
        """
        This method will attempt to fix the mismatches between the column data types and the values within those columns. Fixes include converting:
        1. date/time columns to time objects, and
        2. scientific numbers to floats.
        """
        if len(self.mixed_types) > 0:
            fixed_cols = []
            if verbose:
                print("Attempting fix...")
            for col in self.mixed_types:
                # check if the column contains time related content
                if ("date" in col.lower() or "time" in col.lower()):
                    if convert_datetime:
                        # change data type in col to date/time
                        self.df[col] = pd.to_datetime(self.df[col], format=datetime_format)
                        # removing col from mixed_types list
                        fixed_cols += [col]
                        if verbose:
                            print("%s column changed to Date/Time format" % col)
                    else:
                        pass
                else:
                    # attempting to convert column to numerical format.
                    try:
                        self.df[col] = self.df.loc[:, col].astype('float64')
                        fixed_cols += [col]
                        if verbose:
                            print("%s column changed to float64 format" % col)
                    except Exception as e:
                        if verbose:
                            print(e)

            # updating the mixed_types list
            for col in fixed_cols:
                self.mixed_types.remove(col)

            # printing results
            if verbose:
                if len(self.mixed_types) != 0:
                    print("Remaining columns with unexpected data types are: ", self.mixed_types)
                else:
                    print("Data types in all columns are as expected")

    def get_mixed_columns_checked(self):
        """
        Returns a list of column names that contain mixed data types.
        """
        return self.mixed_columns_checked

    def get_nulls_dropped(self):
        """
        Returns a boolean indicating whether the drop_null_values() function has been called or not.
        """
        return self.nulls_dropped

    def get_columns(self):
        """
        Returns a list of column names in the dataset.
        """
        return self.df.columns

    def insert_col(self, col_num, col_name, col):
        """
        Inserts a column in the dataset.
        Params:
        col_num - the index number of the columns - i.e. where it should be placed.
        col_name - the name of the column
        col - the column data (a list)
        """
        self.df.insert(col_num, col_name, col)
