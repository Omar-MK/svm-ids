"""
Author: Omar M Khalil 170018405

In this file the following tasks are accomplished:
1. The cleaned and encoded binary and multiclass task pre-unsupervised training and testing datasets are loaded. This includes both the datasets with and without augmented features.
2. The train_svm method is called which:
    I. Searches for the best possible SVM model using the passed kernel. The searches proccess involves a nested recursive feature removal across different regularisation intensities using tratified K fold cross-validation to optimise generalisation of the final model.
    II. Tests the final model and saves the results of its experiments.
"""


import pickle
from SVM import train_and_test_svm
from DataTransformation import balance_by_sampling

def train_svm(path, train_n, test_n, class_labels):
    # loading datasets
    train = pickle.load(open(path + train_n, "rb"))
    test = pickle.load(open(path + test_n, "rb"))
    train = balance_by_sampling(train)
    train = train.sample(500)
    # training and testing
    train_and_test_svm(train, train_n, test, class_labels, stochastic=True, path_model='../trainedModels/augmentation_comp_', path_results='../plots/results/pureSvm/')


def main():
    path = "../datasets/transformed/preUnsupervised/"
    krnl = "linear"

    # loading class encoding labels (keys)
    multiclass_labels = pickle.load(open("../datasets/transformed/multiclass_label_encodings.obj", "rb"))
    binary_labels = pickle.load(open("../datasets/transformed/binary_label_encodings.obj", "rb"))
    labels = [binary_labels, multiclass_labels]

    trn_df_paths = ["trainingset_binary.obj",
                    "trainingset_augmented_binary.obj",
                    "trainingset_multiclass.obj",
                    "trainingset_augmented_multiclass.obj"]

    tst_df_paths = ["testingset_binary.obj",
                    "testingset_augmented_binary.obj",
                    "testingset_multiclass.obj",
                    "testingset_augmented_multiclass.obj"]

    i = 0
    for (trn, tst) in zip(trn_df_paths, tst_df_paths):
        train_svm(path, trn, tst, labels[i%2])
        i += 1


if __name__ == "__main__":
    main()
