from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import scipy as sp
from sklearn.base import BaseEstimator, OneToOneFeatureMixin


class BoltzmanMachine(BaseEstimator, OneToOneFeatureMixin):
    """
    https://towardsdatascience.com/hopfield-networks-neural-memory-machines-4c94be821073
    """

    def __init__(self, iterations=1, hidden_units=10):
        self.iterations = iterations
        self.hidden_units = hidden_units

    def energy(self, X_i):
        X_i_bias = np.concatenate([X_i, np.ones(self.hidden_units)])
        return -0.5 * X_i_bias.T @ self.weights @ X_i_bias

    def fit(self, X):
        self.is_fitted_ = True

        X_bias = np.concatenate([X, np.ones((X.shape[0], self.hidden_units))])
        self.coef_ = (1 / X.shape[0]) * X_bias.T @ X_bias
        np.fill_diagonal(self.coef_, 0)

        return self

    def transform(self, X):
        X_new = X.copy()

        for data_i, X_i in enumerate(X):
            X_i_bias = np.concatenate([X_i, np.ones(self.hidden_units)])
            for i in range(self.iterations):
                idx_to_update = np.random.randint(X_i.shape[0])
                activation_weight = self.coef_[idx_to_update, :] @ X_i_bias
                X_new[data_i, idx_to_update] = 2*(activation_weight > 0).astype(int) - 1

        return X_new
