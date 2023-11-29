from __future__ import annotations
import numpy as np
import scipy as sp
from metaheuristic_designer import ObjectiveVectorFunc
from sklearn.metrics import silhouette_score


class KmedioidsObjective(ObjectiveVectorFunc):
    def __init__(self, dataset: np.ndarray, k: int = 3, precompute_dist=True, p=2, metric_fn=None, mode="max"):
        self.dataset = dataset
        self.k = k
        self.precompute_dist = precompute_dist
        if precompute_dist:
            self.dist_mat = sp.spatial.distance_matrix(x=dataset, y=dataset, p=p)
        if metric_fn is None:
            metric_fn = silhouette_score
        self.metric_fn = metric_fn
        self.labels = np.empty((dataset.shape[0],))
        super().__init__(vecsize=k, low_lim=0, up_lim=dataset.shape[0], mode=mode)

    def objective(self, indices):
        score = 0
        values = np.unique(indices)
        if values.size == 1:
            return float("-inf")

        if self.precompute_dist:
            self.labels = np.argmin(self.dist_mat[:, indices], axis=1)
            score = self.metric_fn(self.dataset, self.labels)
        else:
            raise Exception("Not implemented for non precomputed distance matrix.")

        return score
