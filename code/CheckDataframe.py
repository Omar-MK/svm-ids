"""
Author: Omar M Khalil 170018405

This file contains a script which takes in 2 command line arguments:
    1. The path of a csv file or pandas dataframe object save in binary mode with an extension of ".obj".
    2. A boolen (y/n) used to indicate whether to attempt to fix and save the fixed dataframe.
The purpose of this script is to print information about the presence of nan values and mixed columns in the dataset. Moreover, the script is capable of fixing those errors and saving the modified dataset.
"""

import sys
import pickle
from DataWrangler import DataWrangler

def terminate(msg):
    print(msg)
    exit()

def save_dataset(df, fname, path, save_as="csv"):
    fname = fname.replace(".csv", "")
    fname = fname.replace(".obj", "")
    if save_as== "csv":
        df.to_csv(path + fname + ".csv", index=False)
    elif save_as== "obj":
        pickle.dump(df, open(path + fname + ".obj", "wb"))


def main():
    if len(sys.argv) != 1 or len(sys.argv) != 2:
        terminate("This script requires 1 or 2 command line arguments.\nArg 1 = the dataframe path\nArg2 (optional) = (y/n) to fix the Dataframe or not.")

    file_name = sys.argv[0]
    if not file_name.endswith(".csv") and not file_name.endswith(".obj"):
        terminate("This script works with .csv files or pandas dataframe objects saved in binary format with the file extension .obj only!")

    db = None
    f_extension = ''
    if file_name.endswith(".csv"):
        f_extension = ".csv"
        db = DataWrangler("../datasets/transformed/pre-unsupervised/trainingset_augmented_multiclass.csv", encoding = "cp1252", skipspace=True)
    else:
        f_extension = ".obj"
        df = pickle.load(open(file_name, "rb"))
        db.set_df(df)

    # dropping rows with null values
    print("Checking for null values")
    db.drop_null_values(verbose=True)
    # checking if the data types of the columns match the data types of the majority of contained data, if there are "mixed columns", an auto fix is attempted.
    print("Checking data types...")
    if len(db.check_data_types(verbose=True)) > 0:
        if sys.argv[1] == "y":
            db.autofix_mixed_columns(convert_datetime=False, verbose=True)
            save_dataset(db.get_df(), file_name + "_fixed", "", save_as=f_extension)

if __name__ == "__main__":
    main()
