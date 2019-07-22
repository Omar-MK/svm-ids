"""
Author: Omar M Khalil 170018405

This file contains methods used for plotting several chart types used during
exploration

"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


def plotCountBarChart(df, fig_mate, tick_labels='', show=True, save=False):
    """
    This method is used to plot a bar chart that displays counts of
    unique values of a single column in a dataframe.
    This method uses a FigureMate objects to set its title, x-axis,
    y-axis, and save path.
    """
    y_label = fig_mate.get_y()
    x_label = fig_mate.get_x()
    fig = plt.figure()
    ax = fig.gca()
    counts = df[x_label].value_counts()
    counts.plot.bar(ax = ax)
    ax.set_title(fig_mate.get_heading())
    if tickLabels == '':
        ax.set(ylabel=y_label, xlabel=x_label)
    else:
        ax.set(xticklabels=tickLabels, ylabel=y_label, xlabel=x_label)
    plt.tight_layout()
    if save:
        plt.savefig(fig_mate.get_path() + ".png", dpi=300)
    if show:
        plt.show()
    plt.close()


def plotBox(df, input_cols, output_cols, show=True, save=False, path='plots/'):
    """
    This method is used to plot box plots of multiple columns in a dataframe.
    """

    for outCol in output_cols:
        for inCol in input_cols:
            sns.set_style("whitegrid")
            sns.boxplot(inCol, outCol, data=df)
            plt.title('Box plot of ' + inCol + ' Vs ' + outCol)
            plt.xticks(rotation='vertical')
            plt.xlabel(inCol)
            plt.ylabel(outCol)
            plt.tight_layout()
            if save:
                plt.savefig((path + 'boxplot_' + inCol + '_vs_' + outCol).replace(' ', '_') + ".png", dpi=300)
            if show:
                plt.show()
            plt.close()


def plotViolin(df, input_cols, output_cols, show=True, save=False, path='plots/'):
    """
    This method is used to plot violin plots of multiple columns in a dataframe.
    """

    for outCol in output_cols:
        for inCol in input_cols:
            sns.set_style("whitegrid")
            sns.violinplot(inCol, outCol, data=df)
            plt.title('Violin plot of ' + inCol + ' Vs ' + outCol)
            plt.xticks(rotation='vertical')
            plt.xlabel(inCol)
            plt.ylabel(outCol)
            plt.tight_layout()
            if save:
                plt.savefig((path + 'violinplot' + inCol + '_vs_' + outCol).replace(' ', '_') + ".png", dpi=300)
            if show:
                plt.show()
            plt.close()


def plotClusters(df, input_cols, three_D=False, label_prefixes='', title='', show=True, save=False, path='plots/'):
    """
    This method to plots clusters of multidimensional data on 2D plane.
    The method employs PCA to ahcieve this. No categorical features should be provided in the dataframe.
    The data provided should be pre standardised/normalised for optimal performance.
    The class labels should be the last column of the dataframe.
    """

    # checking input_cols columns contain numerical data only
    for col in input_cols:
        type = df[col].dtypes
        if type != "float64" and type != "int64":
            raise ValueError("balance_sample_counts only works with numerical data, and the dataframe passed contains non-numerical inputs.")

    # creating PCA df
    n_dimensions = 2
    if three_D:
        n_dimensions = 3
    y = df.iloc[:, -1]
    pca = PCA(n_components=n_dimensions)
    components = pca.fit_transform(df[input_cols])
    df = None
    if three_D:
        df = pd.concat([pd.DataFrame(data=components, columns=["PC1", "PC2", "PC3"]), y], axis=1)
    else:
        df = pd.concat([pd.DataFrame(data=components, columns=["PC1", "PC2"]), y], axis=1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    if three_D:
        ax = Axes3D(fig)
        ax.set_zlabel("Principla Component 3")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title(title)

    # creating figure
    y_set = set(y)
    colors = ['red', 'blue', 'green', 'orange', 'magenta', 'gray', 'purple', 'brown', 'pink', 'black', 'yellow']
    for (label, color) in zip(y_set, colors):
        indicies = df.iloc[:, -1] == label
        if three_D:
            ax.scatter(df.loc[indicies, "PC1"], df.loc[indicies, "PC2"], df.loc[indicies, "PC3"], c=color, s=50)
        else:
            ax.scatter(df.loc[indicies, "PC1"], df.loc[indicies, "PC2"], c=color, s=50)

    legend_labels = [label_prefixes + str(s) for s in list(y_set)]
    ax.legend(y_set)
    ax.grid()

    if save:
        plt.savefig((path + '_pca_clustering').replace(' ', '_') + ".png", dpi=300)
    if show:
        plt.show()

    plt.close()


def plotSeperation(df, cols, std_dev=0, show=True, save=False, path='plots/'):
    """
    This method is used to visualise the seperation between features for different classes within a dataframe. The method expects the class column to be the last column in the dataframe. The cols parameter should be used to pass the numerical columns in the dataframe.
    """
    i = 0 # index used to change color
    colors = ['red', 'blue', 'green', 'orange', 'magenta', 'gray', 'purple', 'brown', 'pink', 'black', 'yellow', 'cyan']
    fig = plt.figure()
    ax = fig.gca()
    if std_dev == 0:
        ax.set_title('Line plot of feature mean value for each class')
    else:
        ax.set_title('Line plot of feature mean value with ' + str(std_dev) + ' SD error regions for each class')
    ax.set_xlabel('Features')
    ax.set_ylabel('Feature mean value per class')
    for label in set(df.iloc[:, -1]):
        label_rows = df[df.iloc[:, -1] == label]
        means = np.array([])
        stdDevs = np.array([])
        column_ints = np.arange(0, len(cols), 1)
        for col in cols:
            means = np.append(means, label_rows[col].mean())
            if std_dev > 0:
                stdDevs = np.append(stdDevs, label_rows[col].std())

        plt.plot(column_ints, means, 'k', color=colors[i], label='Class: ' + str(label))
        if std_dev > 0:
            plt.plot(column_ints, (means + std_dev * stdDevs), '--', color=colors[i])
            plt.plot(column_ints, (means - std_dev * stdDevs), '--', color=colors[i])
            plt.fill_between(column_ints, (means + std_dev * stdDevs), (means - std_dev * stdDevs), facecolor=colors[i], alpha=0.5)
        i += 1
    ax.legend()
    plt.tight_layout()
    if save:
        fig_name = (path + '_seperability_plot_std_' + str(std_dev).replace('.', '_'))
        plt.savefig(fig_name + ".png", dpi=300)
    if show:
        plt.show()
    plt.close()
