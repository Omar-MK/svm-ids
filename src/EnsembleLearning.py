"""
Author: Omar M Khalil 170018405

In this file the following tasks are accomplished:
1. The cleaned and encoded binary and multiclass task post-unsupervised training and testing datasets are loaded.
2. The train_svm method is called which:
    I. Searches for the best possible SVM model using the passed kernel. The searches proccess involves a nested recursive feature removal across different regularisation intensities using tratified K fold cross-validation to optimise generalisation of the final model.
    II. Tests the final model and saves the results of its experiments.
"""


import pickle
from SVM import train_and_test_svm
from DataTransformation import balance_by_sampling
from sklearn.preprocessing import StandardScaler
import pandas as pd

def scale_train_test(train, test, last_col_index):
    # only scales first col to last col index
    scaler = StandardScaler().fit(train.iloc[:, :last_col_index])
    scaled_data = scaler.transform(train.iloc[:, :last_col_index].values)
    X = pd.DataFrame(scaled_data, index=train.index, columns=train.columns[:last_col_index])
    train = pd.concat([X, train.iloc[:, last_col_index:]], axis=1)
    scaled_data = scaler.transform(test.iloc[:, :last_col_index].values)
    X = pd.DataFrame(scaled_data, index=test.index, columns=train.columns[:last_col_index])
    test = pd.concat([X, test.iloc[:, last_col_index:]], axis=1)
    return train, test

def train_svm(path, train_n, test_n, class_labels):
    # loading datasets
    train = pickle.load(open(path + train_n, "rb"))
    test = pickle.load(open(path + test_n, "rb"))
    train = balance_by_sampling(train)
    test = balance_by_sampling(test)
    train, test = scale_train_test(train, test, -27)
    if len(train) > 5000:
        train = train.sample(5000)
    train_n = train_n.replace("trainingset", '').replace('_', ' ')
    # training and testing
    train_and_test_svm(train, train_n.replace(".obj", ''), test, class_labels, path_model='../trainedModels/ensemble_model_', path_results='../plots/results/ensembleLearning/')


def main():
    path = "../datasets/transformed/postUnsupervised/"
    path_suff = ["_Pearson_R_Reduction_(PRR)_",
                 "_PRR_+_Clustering_"]

    # loading class encoding labels (keys)
    multiclass_labels = pickle.load(open("../datasets/transformed/multiclass_label_encodings.obj", "rb"))
    binary_labels = pickle.load(open("../datasets/transformed/binary_label_encodings.obj", "rb"))
    labels = [binary_labels, multiclass_labels]

    trn_df_paths = ["trainingset_augmented_binary" + path_suff[1] + ".obj",
                    "trainingset_augmented_multiclass" + path_suff[1] +  ".obj"]

    tst_df_paths = ["testingset_augmented_binary" + path_suff[1] + ".obj",
                    "testingset_augmented_multiclass" + path_suff[1] +  ".obj"]

    i = 0
    for (trn, tst) in zip(trn_df_paths, tst_df_paths):
        train_svm(path, trn, tst, labels[i%2])
        i += 1


if __name__ == "__main__":
    main()
