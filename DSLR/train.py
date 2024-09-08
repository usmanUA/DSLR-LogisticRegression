# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    train.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: uahmed <uahmed@student.hive.fi>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/09/06 22:36:50 by uahmed            #+#    #+#              #
#    Updated: 2024/09/07 22:28:26 by uahmed           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from DSLR.model import LogisticRegression
from DSLR.utils import modelData, plotErrorCost
from DSLR.preprocessing import trainTestSplit, Standardizer
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

def trainModel(dataset, features):
    '''
    Train a Logistic Regression model and saves the weights to a file
    Parameters
    ----------
    data: the dataset to train the model for
    '''
    X, y = modelData(dataset=dataset, features=features, action='train')
    X_train, X_test, y_train, y_test = trainTestSplit(X, y, testSize=0.3, randomState=4)

    sc = Standardizer()
    sc.fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)

    lr = LogisticRegression(eta=0.01, maxIter=100, Lambda=10)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    lr.save_weights(sc)
    print(f'Incorrectly Classified Samples: {sum(y_test != y_pred)}')
    print(f'Accuracy Score: {accuracy_score(y_test, y_pred):.2f}')

    plotErrorCost(lr._cost, lr._errors)

