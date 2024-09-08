# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    predict.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: uahmed <uahmed@student.hive.fi>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/09/07 21:57:24 by uahmed            #+#    #+#              #
#    Updated: 2024/09/07 22:42:52 by uahmed           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
from DSLR.utils import modelData
from DSLR.model import LogisticRegression
from DSLR.preprocessing import Standardizer
from sklearn.metrics import accuracy_score

def predict(dataset, features, filename):
    '''
    Predicts the Houses given the dataset.
    Uses the learned weights save into the file.
    Parameters
    ----------
    data: the dataset to train the model for
    filename: name of the file containing learned weights
    '''
    X, y = modelData(dataset, features, 'predict')
    df = pd.read_csv(filename)
    K = list(df)[:4]
    mean = df.values[1:, 4]
    std = df.values[1:, 5]
    weights = df.values[:, :4].T

    sc = Standardizer(mean=mean, std=std)
    X = sc.transform(X)

    lr = LogisticRegression(initial_weights=weights, classes=K)
    preds = lr.predict(X)
    with open('datasets/houses.csv', 'w') as f:
        f.write('Index,Hogwarts House\n')
        for i in range(len(preds)):
            f.write(f'{i},{preds[i]}\n')
