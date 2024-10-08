# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    utils.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: uahmed <uahmed@student.hive.fi>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/09/07 22:37:31 by uahmed            #+#    #+#              #
#    Updated: 2024/09/07 22:37:32 by uahmed           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


importantFeatures = ['Astronomy', 'Herbology', 'Divination', 'Ancient Runes', 'Charms', 'Flying']

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

def modelData(dataset, features, action):
    '''
    Parses data selecting the features for the training/predictions.
    Parameters
    ----------
    dataset: numpy ndarray of the given dataset

    Returns
    ------
    X: selected features
    y: feature to learn/predict
    '''
    df = pd.DataFrame(dataset, columns=features)
    if action == 'train':
        df = df.dropna(subset=importantFeatures)
    else:
        df = df.fillna(method='ffill')
    y = df.values[:, 0]
    dfX = df[importantFeatures]
    X = dfX.to_numpy(dtype=float)
    return X, y

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


def plotErrorCost(cost, error):
    '''
    Plots the cost and error of the Logistic Regression Model.
    Parameters
    ---------
    cost: Cost History, shape (n_iter, )
    error: Errors History , shape (n_iter, )
    '''

    _, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6), constrained_layout=True)
    ax[0].plot(range(1, len(cost)+1), cost, marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Cost Function')
    ax[0].set_title('Logistic Regression - Learning Rate 0.1 / Regularizatioin term 10')

    ax[1].plot(range(1, len(error)+1), error, marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Missclasifications')
    ax[1].set_title('Logistic Regression - Learning Rate 0.1 / Regularizatioin term 10')

    plt.show()
