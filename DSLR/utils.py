import csv
import numpy as np
import matplotlib.pyplot as plt


def loadDataset(fileName):
    '''Loads the dataset file and writes entries to a numpy array'''
    dataset = []
    with open(fileName, 'r') as file:
        reader = csv.reader(file)
        try:
            for rw in reader:
                row = []
                for entry in rw:
                    try:
                        entry = float(entry)
                    except:
                        if not entry:
                            entry = np.nan
                    row.append(entry)
                dataset.append(row)
        except csv.Error as e:
            print(f"File: {fileName}, line number: {reader.line_num}, error: {e}")
    return  np.array(dataset, dtype=object)

def getDataFeatures(dataset):
    ''' Returns the features names of the given dataset (column names)'''
    return dataset[0, 1:]

def getData(dataset):
    ''' Returns the data of the given dataset (excluding the columns names)'''
    return dataset[1:, 1:]

def getLegends(data, index):
    '''Returns the legends for the plot'''

    sortedData = data[data[:, index].argsort()]
    hogHouse = sortedData[:, index]
    legend = sorted(set(hogHouse))
    return legend, hogHouse, sortedData

def getParameters(features, data):
    '''Parses parameters for histogram plot'''
    title = features[15]
    legend, hogHouse, sortedData = getLegends(data, 0)
    elems, inds = np.unique(hogHouse, return_index=True)
    indices = {}
    i = 0
    for ind, elem in enumerate(elems):
        indices[legend[i]] = int(inds[ind])
        i += 1
    X = sortedData[:, 15]
    X = X.astype("float")
    X = X[~np.isnan(X)]
    return X, title, legend, indices

def plotGraph(X, Y, legend, indices, ax=None):
    '''Plots the graph based on the given instruction in the parameters'''

    tot = len(legend)
    colors = ['red', 'yellow', 'blue', 'green']
    for i in range(0, tot):
        color = colors[i]
        if i == tot - 1:
            if Y is None:
                h = X[indices[legend[i]]:]
                h = h[~np.isnan(h)]
                if ax is None:
                    plt.hist(h, color=color, alpha=0.5)
                else:
                    ax.hist(h, alpha=0.5)
            else:
                x = X[indices[legend[i]]:]
                y = Y[indices[legend[i]]:]
                if ax is None:
                    plt.scatter(x, y, color=color, alpha=0.5)
                else: 
                    ax.scatter(x, y, s=1, color=color, alpha=0.5)
        else:
            if Y is None:
                h = X[indices[legend[i]]:indices[legend[i+1]]]
                h = h[~np.isnan(h)]
                if ax is None:
                    plt.hist(h, color=color, alpha=0.5)
                else: 
                    ax.hist(h, alpha=0.5)
            else:
                x = X[indices[legend[i]]:indices[legend[i+1]]]
                y = Y[indices[legend[i]]:indices[legend[i+1]]]
                if ax is None:
                    plt.scatter(x, y, color=color, alpha=0.5)
                else:
                    ax.scatter(x, y, s=1, color=color, alpha=0.5)
