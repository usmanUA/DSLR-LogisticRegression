# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    pairplot.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: uahmed <uahmed@student.hive.fi>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/09/02 10:48:52 by uahmed            #+#    #+#              #
#    Updated: 2024/09/02 12:05:37 by uahmed           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import matplotlib.pyplot as plt
from DSLR.utils import plotGraph, getParameters
import numpy as np

def pairplot(features, data):
    '''Plots pair plot based on the given data'''

    font = {
        'family': 'DejaVu Sans',
        'weight': 'light',
        'size': 7
    }
    plt.rc('font', **font)
    _, _, legend, indices = getParameters(features, data)
    features = features[5:]
    data = np.array(data[:, 5:], dtype=float)

    size = data.shape[1]
    fig, axs = plt.subplots(nrows=size, ncols=size, figsize=(25.6, 14.4), tight_layout=False)
    plt.subplots_adjust(wspace=0.15, hspace=0.15)
    for row in range(size):
        for col in range(size):
            ax = axs[row, col]
            X = data[:, col]
            if row == col:
                plotGraph(X, None, legend, indices, ax)
            else:
                Y = data[:, row]
                plotGraph(X, Y, legend, indices, ax)
            ax.tick_params(labelbottom=False)
            ax.tick_params(labelleft=False)
            if row == size - 1:
                ax.set_xlabel(features[col].replace(' ', '\n'))
            if col == 0:
                feature = features[col].replace(' ', '\n')
                length = len(feature)
                if length > 14 and '\n' not in feature:
                    feature = feature[:int(length/2)] + '\n' + feature[int(length/2):]
                ax.set_ylabel(feature)

    plt.legend(legend, loc='center left', frameon=False, bbox_to_anchor=(1, 0.5))
    plt.show()


