# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    model.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: uahmed <uahmed@student.hive.fi>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/09/05 08:32:42 by uahmed            #+#    #+#              #
#    Updated: 2024/09/07 22:34:45 by uahmed           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
from numpy._core.defchararray import index


class   LogisticRegression(object):
    '''Trains the model based on the logistic regression algorithm'''

    def __init__(self, eta=0.1, Lambda=0, maxIter=50, initial_weights=None, classes=None) -> None:
        self.eta = eta
        self.Lambda = Lambda
        self.maxIter = maxIter
        self._weights = initial_weights
        self._K = classes
        self._cost = []
        self._errors = []

    def fit(self, X, y, weights=None):
        '''
        Fits the model and returns the optimized parameters
        Parameters
        ----------
        X: array-like, shape [n_samples, n_features]
        y: array, shape [n_features]
        '''

        newX = np.insert(X, 0, 1, axis=1)
        m = newX.shape[0]
        self._K = np.unique(y).tolist()
        self._weights = weights
        if not self._weights:
            self._weights = np.zeros((newX.shape[1] * len(self._K)))
        self._weights = self._weights.reshape(len(self._K), newX.shape[1])

        yVec = np.zeros((len(y), len(self._K)))
        for i in range(len(y)):
            yVec[i, self._K.index(y[i])] = 1

        for _ in range(self.maxIter):
            preds = self.sigmoid(self._weights.dot(newX.T)).T
            yOneTerm = yVec.T.dot(np.log(preds))
            yZeroTerm = (1 - yVec).T.dot(np.log(1 - preds))
            r1 = (self.Lambda / (2 * m)) * sum(sum(self._weights[:, 1:] ** 2))
            #print(r1)
            cost = (-1 / m) * sum(yOneTerm + yZeroTerm) + r1
            self._cost.append(cost)
            self._errors.append(sum(y != self.predict(X)))
            r2 = (self.Lambda / m) * self._weights[:, 1:]
            self._weights = self._weights - (self.eta * (1 / m) * (preds - yVec).T.dot(newX) + np.insert(r2, 0, 0, axis=1))

#        print(self._cost)
        return self

    def predict(self, X):
        '''
        Predicts the outcome based on the learned weights using X data
        Parameters
        ----------
        X: numpy ndarray, shape [n_samples, n_features]

        Returns
        -------
        Predicted outcome
        '''
        X = np.insert(X, 0, 1, axis=1)
        preds = self.sigmoid(self._weights.dot(X.T)).T
        return [self._K[x] for x in preds.argmax(1)]

    def sigmoid(self, z):
        '''
        Hypothesis function for Logistic Regression
        Parameters
        ----------
        z: theta.T * X

        Returns
        -------
        Probabilities for the classes
        '''
        return 1.0 / (1.0 + np.exp(-z))

    def save_weights(self, sc, filename='./datasets/weights.csv'):
        '''
        Saves the optimized weights to a file.
        Parameters
        ----------
        sc: Standardizer object to calculate mean, std etc.
        filename: Name of the file to save the weights to.
        
        Returns
        -------
        self: Returns the object
        '''

        with open(filename, 'w') as f:
            for i in range(len(self._K)):
                f.write(f'{self._K[i]},')
            f.write('Mean,Std\n')
            for j in range(0, self._weights.shape[1]):
                for i in range(0, self._weights.shape[0]):
                    f.write(f'{self._weights[i][j]},')
                f.write(f'{sc._mean[j - 1]},{sc._std[j - 1]}\n')
        return self
