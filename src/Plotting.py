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
from textwrap import wrap

def construct_frequency_plot(df, col, fm, show=1, save=0):
    """
    This method is used to plot a bar chart that displays counts of
    unique values of a single column in a dataframe.
    This method uses a FigureMate object to set its title, x-axis,
    and y-axis prefixes, and save path.
    """
    x_label = fm.x_lbl
    y_label = fm.y_lbl
    ts = fm.heading
    if fm.prefix:
        x_label += ' ' + str(col)
        y_label += " Frequency "
        ts += " Frequency of " + x_label + 's'

    fig = plt.figure()
    ax = fig.gca()
    counts = df[col].value_counts()
    counts.plot.bar(ax = ax)
    title = ax.set_title("\n".join(wrap(ts.title(), 60)))
    if fm.tick_labels is None:
        ax.set(ylabel=y_label, xlabel=x_label)
    else:
        ax.set(xticklabels=fm.tick_labels, ylabel=y_label, xlabel=x_label)
    finalise_figure(title, fig, plt, fm.path, ts, save, show)


def construct_box_plot(df, input_cols, output_cols, fm, show=1, save=0):
    """
    This method is used to plot box plots of multiple columns in a dataframe.
    This method uses a FigureMate objects to set the figure title, x-axis, and
    y-axis prefixes, as well as the save path.
    """
    for y in output_cols:
        for x in input_cols:
            x_label = fm.x_lbl
            y_label = fm.y_lbl
            ts = fm.heading
            if fm.prefix:
                x_label += ' ' + x
                y_label += ' ' + y
                ts += " Box plot of " + x_label + " vs " + y_label
            sns.set_style("whitegrid")
            sns.boxplot(x, y, data=df)
            title = plt.title("\n".join(wrap(ts.title(), 60)))
            plt.xticks(rotation='vertical')
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            finalise_figure(title, plt, plt, fm.path, ts, save, show)


def construct_violin_plot(df, input_cols, output_cols, fm, show=1, save=0):
    """
    This method is used to plot violin plots of multiple columns in a dataframe.
    This method uses a FigureMate objects to set the figure title, x-axis, and
    y-axis prefixes, as well as the save pat
    """
    for y in output_cols:
        for x in input_cols:
            x_label = fm.x_lbl
            y_label = fm.y_lbl
            ts = fm.heading
            if fm.prefix:
                x_label += ' ' + x
                y_label += ' ' + y
                ts += " Violin plot of " + x_label + " vs " + y_label
            sns.set_style("whitegrid")
            sns.violinplot(x, y, data=df)
            title = plt.title("\n".join(wrap(ts.title(), 60)))
            plt.xticks(rotation='vertical')
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            finalise_figure(title, plt, plt, fm.path, ts, save, show)


def construct_cluster_plot(df, input_cols, fm, dimensions=2, show=1, save=0):
    """
    This method to plots clusters of multidimensional data on a 2D/3D plane
    based on the number passed in the dimensions parameter. Default value = 2,
    any value passed other than 3 and the method will default to 2D plots.
    The method employs PCA to ahcieve this. No categorical features should be
    provided in the dataframe. The data provided should also be pre-normalised
    for optimal performance. Moreover, the class labels should be the last
    column of the dataframe.
    This method uses a FigureMate objects to set the figure title, x-axis, and
    y-axis prefixes, as well as the save path.
    """
    # checking input_cols columns contain numerical data only
    for col in input_cols:
        type = df[col].dtypes
        if not np.issubdtype(type, np.number):
            raise ValueError("balance_sample_counts only works with numerical data, and the dataframe passed contains non-numerical inputs.")

    # checking number of dimensions to plot
    if dimensions != 2 and dimensions != 3:
        dimensions = 2

    # creating PCA df
    y = df.iloc[:, -1]
    pca = PCA(n_components=dimensions)
    components = pca.fit_transform(df[input_cols])
    if dimensions == 3:
        df = pd.concat([
            pd.DataFrame(
                data=components, columns=["PC1", "PC2", "PC3"]), y], axis=1)
    else:
        df = pd.concat([
            pd.DataFrame(data=components, columns=["PC1", "PC2"]), y], axis=1)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    if dimensions == 3:
        ax = Axes3D(fig)
        ax.set_zlabel("PC 3")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ts = fm.heading
    if fm.prefix:
        ts += " Cluster diagram"
    title = ax.set_title("\n".join(wrap(ts.title(), 60)))

    # creating figure
    y_set = set(y)
    colors = ['red', 'blue', 'green', 'orange', 'magenta', 'gray', 'purple', 'brown', 'pink', 'black', 'yellow']
    for (label, color) in zip(y_set, colors):
        indicies = df.iloc[:, -1] == label
        if dimensions == 3:
            ax.scatter(df.loc[indicies, "PC1"], df.loc[indicies, "PC2"], df.loc[indicies, "PC3"], marker='x', c=color, s=8)
        else:
            ax.scatter(df.loc[indicies, "PC1"], df.loc[indicies, "PC2"],  marker='x', c=color, s=8, alpha=0.35)

    if fm.legend_labels is None:
        ax.legend(list(y_set), loc='best', fontsize='small')
    else:
        ax.legend(fm.legend_labels, loc='best', fontsize='small')
    ax.grid()
    finalise_figure(title, fig, plt, fm.path, ts, save, show)


def construct_seperation_plot(df, cols, fm, std_dev=0, show=1, save=0):
    """
    This method is used to visualise the seperation between features for
    different classes within a dataframe. The method expects the class column
    to be the last column in the dataframe. The cols parameter should be used
    to pass the numerical columns in the dataframe.
    """
    i = 0 # index used to change color
    colors = ['red', 'blue', 'green', 'orange', 'magenta', 'gray', 'purple', 'brown', 'pink', 'black', 'yellow', 'cyan']
    fig = plt.figure()
    ax = fig.gca()
    ts = fm.heading
    if fm.prefix:
        if std_dev == 0:
            ts += "Line plot of feature mean value per class"
        else:
            ts += "Line plot of feature mean value with " + str(std_dev) + " SD error regions for each class"
    title = ax.set_title("\n".join(wrap(ts.title(), 60)))
    ax.set_xlabel('Feature')
    ax.set_ylabel('Feature mean value')
    for label in set(df.iloc[:, -1]):
        label_rows = df[df.iloc[:, -1] == label]
        means = np.array([])
        stdDevs = np.array([])
        column_ints = np.arange(0, len(cols), 1)
        for col in cols:
            means = np.append(means, label_rows[col].mean())
            if std_dev > 0:
                stdDevs = np.append(stdDevs, label_rows[col].std())
        if fm.legend_labels is None:
            plt.plot(column_ints, means, 'k', color=colors[i], label='Class: ' + str(label))
        else:
            plt.plot(column_ints, means, 'k', color=colors[i], label=fm.legend_labels[i])
        if std_dev > 0:
            plt.plot(column_ints, means, '--', color=colors[i])
            plt.fill_between(column_ints,
                (means + std_dev * stdDevs),
                (means - std_dev * stdDevs),
                facecolor=colors[i],
                alpha=0.35)
        i += 1
    ax.legend(loc='best', fontsize='small')
    finalise_figure(title, fig, plt, fm.path, ts, save, show)


def finalise_figure(title, fig, plt, path, ts, save, show):
    fig.tight_layout()
    title.set_y(1.05)
    fig.subplots_adjust(top=0.8)
    if save:
        plt.savefig((path + ts + ".png").replace(' ', '_'), dpi=500)
    if show:
        plt.show()
    plt.close()
