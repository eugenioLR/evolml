from __future__ import annotations
import numpy as np
import scipy as sp
from metaheuristic_designer import VectorObjectiveFunc
from sklearn.metrics import silhouette_score


class KmedoidsObjective(VectorObjectiveFunc):
    def __init__(self, k: int = 3, precompute_dist=True, p=2, metric_fn=None, dataset: np.ndarray = None, mode="max"):
        self.k = k
        self.p = p
        self.precompute_dist = precompute_dist

        up_lim = None
        if dataset is not None:
            if precompute_dist:
                self.dist_mat = sp.spatial.distance_matrix(x=dataset, y=dataset, p=p)
            up_lim = dataset.shape[0]
        self.dataset = dataset

        if metric_fn is None:
            metric_fn = silhouette_score
        self.metric_fn = metric_fn

        super().__init__(vecsize=k, low_lim=0, up_lim=up_lim, mode=mode, name="Kmedoids")

    def set_data(self, dataset):
        self.dataset = dataset
        if precompute_dist:
            self.dist_mat = sp.spatial.distance_matrix(x=dataset, y=dataset, p=p)
        self.up_lim = dataset.shape[0]

    def compute_distance(self, medioids, X):
        return sp.spatial.distance_matrix(x=medioids, y=X, p=self.p)

    def get_clusters(self, indices):
        if self.precompute_dist:
            labels = np.argmin(self.dist_mat[:, indices], axis=1)
        else:
            raise Exception("Not implemented for non precomputed distance matrix.")

        return labels

    def objective(self, indices):
        score = 0
        values = np.unique(indices)
        if values.size == 1:
            return -float("inf")

        labels = self.get_clusters(indices)
        score = self.metric_fn(self.dataset, labels)

        return score
