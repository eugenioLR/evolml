from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple
import torch
import scipy as sp
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LinearRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.cluster import KMeans


class RBFNNModel(ABC, BaseEstimator, ClassifierMixin):
    """ """

    def __init__(
        self,
        n_units: int,
        linear_layer: BaseEstimator = None,
        cluster_model: BaseEstimator = None,
        std_from_clusters: bool = False,
        classification: bool = False,
        random_state: int = None,
    ):
        self.n_units = n_units

        self.linear_layer = linear_layer

        if cluster_model is None:
            cluster_model = KMeans(n_clusters=n_units, random_state=random_state)
        self.cluster_model = cluster_model

        self.std_from_clusters = std_from_clusters

        self.classification = classification

        self.random_state = random_state

    def _fit_clustering(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Cluster input data into 'n_units' clusters
        self.cluster_model = self.cluster_model.fit(X)
        cluster_idx = self.cluster_model.predict(X)

        # Group points by cluster
        cluster_idx_sorted = cluster_idx.copy()
        cluster_idx_sorted.sort()
        _, cluster_idx_pos = np.unique(cluster_idx_sorted, return_index=True)
        X_sorted = X[cluster_idx.argsort()]
        X_grouped = np.split(X_sorted, cluster_idx_pos[1:])

        # Obtain centers and widths
        centers = self.cluster_model.cluster_centers_
        widths = np.empty(self.n_units)
        for idx, X_cluster in enumerate(X_grouped):
            if self.std_from_clusters:
                widths[idx] = 1 / X_cluster.var()
            else:
                distance_to_centroid = sp.spatial.distance.cdist(X_cluster, centers[[idx], :])
                widths[idx] = np.sqrt(2 * self.n_units) / distance_to_centroid.max()

        return centers, widths

    def _rbf_layer(self, X: np.ndarray, centers: np.ndarray, widths: np.ndarray) -> np.ndarray:
        """ """

        distances = sp.spatial.distance.cdist(X, centers) ** 2
        X_rbf = np.exp(-distances * widths)
        return X_rbf

    def fit(self, X: np.ndarray, y: np.ndarray) -> RBFNNClassifier:
        """ """

        X, y = check_X_y(X, y)
        if self.classification:
            self.classes_ = unique_labels(y)

        self.centers_, self.widths_ = self._fit_clustering(X)
        X_rbf = self._rbf_layer(X, self.centers_, self.widths_)
        self.linear_layer = self.linear_layer.fit(X_rbf, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """ """

        check_is_fitted(self)
        X = check_array(X)

        X_rbf = self._rbf_layer(X, self.centers_, self.widths_)

        return self.linear_layer.predict(X_rbf)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """ """

        check_is_fitted(self)
        X = check_array(X)

        X_rbf = self._rbf_layer(X, self.centers_, self.widths_)

        return self.linear_layer.predict_proba(X_rbf)


class RBFNNClassifier(RBFNNModel):
    """ """

    def __init__(
        self,
        n_units: int,
        probability: float = True,
        linear_layer: BaseEstimator = None,
        cluster_model: BaseEstimator = None,
        std_from_clusters: bool = False,
        random_state: int = None,
    ):
        self.probability = probability

        if linear_layer is None:
            if probability:
                linear_layer = LogisticRegression(penalty=None, random_state=random_state)
            else:
                linear_layer = RidgeClassifier(alpha=0, solver="svd", random_state=random_state)

        super().__init__(
            n_units=n_units,
            linear_layer=linear_layer,
            cluster_model=cluster_model,
            std_from_clusters=std_from_clusters,
            classification=True,
            random_state=random_state,
        )


class RBFNNRegressor(RBFNNModel):
    """ """

    def __init__(
        self,
        n_units: int,
        linear_layer: BaseEstimator = None,
        cluster_model: BaseEstimator = None,
        std_from_clusters: bool = False,
        random_state: int = None,
    ):
        if linear_layer is None:
            linear_layer = LinearRegression()

        super().__init__(
            n_units=n_units,
            linear_layer=linear_layer,
            cluster_model=cluster_model,
            std_from_clusters=std_from_clusters,
            classification=False,
            random_state=random_state,
        )
