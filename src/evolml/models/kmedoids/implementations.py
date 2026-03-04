from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import scipy as sp
from sklearn.base import BaseEstimator, ClusterMixin
from metaheuristic_designer.algorithms import GeneralAlgorithm
from metaheuristic_designer.strategies import GA, HillClimb
from metaheuristic_designer.initializers import UniformInitializer
from metaheuristic_designer.operators import VectorOperator
from metaheuristic_designer.selection_methods import ParentSelection, SurvivorSelection
from .Kmedoids_objective import KmedoidsObjective


class BaseKMedoids(ABC, BaseEstimator, ClusterMixin):
    def __init__(self, k=3, **kwargs):
        self.k = k
        self.medioids = None
        self.precompute_dist = kwargs.get("precompute_dist", True)
        self.metric_p = kwargs.get("metric_p", 2)
        self.metric_fn = kwargs.get("metric_fn", None)
        self.objfunc = None
        self.optimizer_params = kwargs

    @abstractmethod
    def fit(self, X, _y=None):
        """
        Find the medioids given a dataset
        """

    def predict(self, X):
        dist_mat = self.objfunc.compute_distance(X, self.medioids)
        return np.argmin(dist_mat, axis=1)


class GreedyKMedoids(BaseKMedoids):
    def __init__(self, k=3, **kwargs):
        super().__init__(k, **kwargs)

    def fit(self, X, y=None):
        self.objfunc = KmedoidsObjective(dataset=X, k=self.k, precompute_dist=self.precompute_dist, p=self.metric_p, metric_fn=self.metric_fn)
        initializer = UniformInitializer(self.k, 0, X.shape[0] - 1, pop_size=1, dtype=int)

        strategy = HillClimb(
            initializer,
            operator=VectorOperator("MutSample", {"distrib": "Uniform", "min": 0, "max": X.shape[0] - 1, "N": 1}),
        )

        algorithm = GeneralAlgorithm(self.objfunc, strategy, params=self.optimizer_params)

        result = algorithm.optimize()
        best_solution, best_fitness = result.best_solution()
        self.medioids = X[best_solution, :]

        return self


class GeneticKMedoids(BaseKMedoids):
    def __init__(self, k=3, **kwargs):
        super().__init__(k, **kwargs)
        self.pcross = kwargs.get("pcross", 0.9)
        self.pmut = kwargs.get("pmut", 0.1)
        self.pop_size = kwargs.get("pop_size", 100)

    def fit(self, X, _y=None):
        self.objfunc = KmedoidsObjective(dataset=X, k=self.k, precompute_dist=self.precompute_dist, p=self.metric_p, metric_fn=self.metric_fn)
        initializer = UniformInitializer(self.k, 0, X.shape[0] - 1, pop_size=self.pop_size, dtype=int)

        strategy = GA(
            initializer,
            cross_op=VectorOperator("Multipoint"),
            mutation_op=VectorOperator("MutSample", {"distrib": "uniform", "min": 0, "max": X.shape[0] - 1, "N": 1}),
            parent_sel=ParentSelection("Tournament", {"amount": 3, "p": 0.8}),
            survivor_sel=SurvivorSelection("KeepBest"),
            params={"pcross": self.pcross, "pmut": self.pmut},
        )

        algorithm = GeneralAlgorithm(self.objfunc, strategy, params=self.optimizer_params)

        result = algorithm.optimize()
        best_solution, best_fitness = result.best_solution()
        self.medioids = X[best_solution, :]

        return self
