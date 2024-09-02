# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    scatterplot.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: uahmed <uahmed@student.hive.fi>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/09/01 20:56:41 by uahmed            #+#    #+#              #
#    Updated: 2024/09/01 20:59:43 by uahmed           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import matplotlib.pyplot as plt
from DSLR.utils import getParameters, plotGraph

def scatterplot(features, data):
    '''Plots the Scatter Plot based on the two similar features'''

    xIdx = 6
    yIdx = 8
    _, _, legend, indices = getParameters(features, data)
    X = np.array(data[:,xIdx], dtype=float)
    y = np.array(data[:,yIdx], dtype=float)
    plotGraph(X, y, legend, indices)
    xlabel = data[0, xIdx]
    ylabel = data[0, yIdx]
#    plt.scatter(X, y)# legend=legend)
    plt.xlabel(xlabel)
    plt.xlabel(ylabel)
    plt.legend(legend, loc='upper right', frameon=False)
    plt.show()
