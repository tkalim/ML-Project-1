# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    feature_names = np.genfromtxt(data_path, delimiter=",", skip_footer=x.shape[0], dtype=str)[2:]
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids, feature_names


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
            

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-45, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def popCorrelatedFeatures(X, threshold):
    corMat = np.abs(np.corrcoef(X.T))
    
    new_features = set(range(corMat.shape[0]))
    
    indices = np.where(corMat >= threshold)
    
    tab=[]
    
    for n in range(len(indices[0])):
        i, j = indices[0][n], indices[1][n]
        if(i>j):
            continue
        if(i==j):
            tab.append([i])
        else:
            tab[i].append(j)
        
    for k in range(len(tab)):
        if((k not in new_features)):
            continue
            
        if(len(tab[k]) != 1):
            for p in range(1, len(tab[k])):
                new_features.remove(tab[k][p])
    
    res = np.zeros(X.shape[0]).T
    
    for k in range(corMat.shape[0]):
        if(k in new_features):
            res = np.column_stack((res, X[:, k]))
            
    res = np.delete(res, 0, 1)
    return res, new_features

def plotHeatMap(matrix, labels_horiz, labels_vert, name, size):
    fig, ax = plt.subplots()

    fig.set_size_inches(size)

#     label_horiz / label_vert order to be checked
    im, cbar = heatmap(matrix, labels_horiz, labels_vert, ax=ax,
                   cmap="YlGn", cbarlabel="Correlation")
    texts = annotate_heatmap(im, valfmt="{x:.2f}")

    fig.tight_layout()
    fig.savefig(name)
    plt.show()
    
    
def plotDistributions(X1, names=[]):
    n = X1.shape[1]
    fig, ax = plt.subplots(nrows=int(n/2), ncols=2)
    fig.set_size_inches((20,100))
    
    nans = np.count_nonzero(np.isnan(X1), axis=0)
    if(len(names)==0):
        for i in range(0,n,2):
            if(nans[i] == X1.shape[0]):
                continue
                
            ax[int(i/2), 0].hist(X1[:, i], bins=250)
            ax[int(i/2), 0].set_ylabel("Number of samples")

            ax[int(i/2), 1].hist(X1[:, i+1], bins=250)
            ax[int(i/2), 1].set_ylabel("Number of samples")
        
    else:
        for i in range(0,n,2):
            if(nans[i] == X1.shape[0]):
                continue
            
            ax[int(i/2), 0].hist(X1[:, i], bins=250)
            ax[int(i/2), 0].set_ylabel("Number of samples")
            ax[int(i/2), 0].set_xlabel(names[i])

            ax[int(i/2), 1].hist(X1[:, i+1], bins=250)
            ax[int(i/2), 1].set_ylabel("Number of samples")
            ax[int(i/2), 1].set_xlabel(names[i+1])
        
    plt.show()
    
def plotNaNpercentage(X, names):
    tmp = np.count_nonzero(np.isnan(X), axis=0)
    
    percentages = tmp*100/(X.shape[0])
    
    fig, ax = plt.subplots()
    fig.set_size_inches((20,15))
    
    ax.barh(names, percentages)
    
    fig.savefig('nanpercentage.png')
    plt.show()
    
    
    
    
    
    
    