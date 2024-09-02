# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    histogram.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: uahmed <uahmed@student.hive.fi>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/09/01 15:09:52 by uahmed            #+#    #+#              #
#    Updated: 2024/09/02 11:57:39 by uahmed           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import matplotlib.pyplot as plt
from DSLR.utils import getParameters, plotGraph

def histogram(features, data, xlabel, ylabel):
    '''Plots Histogram given the feature and necessary information'''
    X, title, legend, indices = getParameters(features, data)
    plotGraph(X, None, legend, indices)
    plt.legend(indices.keys(), loc='upper right', frameon=False)
    plt.xlabel(xlabel=xlabel)
    plt.ylabel(ylabel=ylabel)
    plt.title(title)
    plt.show()
