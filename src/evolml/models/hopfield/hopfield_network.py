from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import scipy as sp
from sklearn.base import BaseEstimator, OneToOneFeatureMixin


class BernoulliHopfieldNetwork(BaseEstimator, OneToOneFeatureMixin):
    """
    https://towardsdatascience.com/hopfield-networks-neural-memory-machines-4c94be821073
    """

    def __init__(self, iterations=1, verbose=False):
        self.iterations = iterations
        self.verbose = verbose

    def energy(self, X_i):
        return -0.5 * np.einsum('ni,ji,nj->n', X_i, self.coef_, X_i)

    def fit(self, X):
        self.is_fitted_ = True

        self.coef_ = (1 / X.shape[0]) * X.T @ X
        np.fill_diagonal(self.coef_, 0)

        return self

    def _update_network(self, X):
        X_new = X

        for data_i, X_i in enumerate(X):
            idx_to_update = np.random.randint(X_i.shape[0])
            activation_weight = self.coef_[idx_to_update, :] @ X_i
            X_new[data_i, idx_to_update] = 2*(activation_weight > 0).astype(int) - 1
        
        return X_new

    def transform(self, X):
        X_new = X.copy()

        for _ in range(self.iterations):
            if self.verbose:
                print(self.energy(X_new))
            X_new = self._update_network(X_new)

        return X_new