from __future__ import annotations
import numpy as np
import scipy as sp
from sklearn.base import BaseEstimator, ClusterMixin
from metaheuristic_designer.algorithms import GeneralAlgorithm
from metaheuristic_designer.strategies import GA
from metaheuristic_designer.initializers import UniformVectorInitializer
from metaheuristic_designer.operators import OperatorInt
from metaheuristic_designer.selectionMethods import ParentSelection, SurvivorSelection
from .Kmedoids_objective import KmedoidsObjective


class GeneticKMedoids(BaseEstimator, ClusterMixin):
    def __init__(self, k=3, **kwargs):
        self.k = k
        self.medioids = None
        self.genetic_params = kwargs
        self.pcross = kwargs.get("pcross", 0.9)
        self.pmut = kwargs.get("pmut", 0.1)
        self.pop_size = kwargs.get("pop_size", 100)
        self.objfunc = None

    def fit(self, X, _y=None):
        self.objfunc = KmedoidsObjective(X, k=self.k)
        initializer = UniformVectorInitializer(self.k, 0, X.shape[0] - 1, pop_size=self.pop_size, dtype=int)

        strategy = GA(
            initializer,
            cross_op=OperatorInt("multipoint"),
            mutation_op=OperatorInt("mutsample", {"distrib": "uniform", "min": 0, "max": X.shape[0] - 1, "N": 1}),
            parent_sel=ParentSelection("Tournament", {"amount": 3, "p": 0.8}),
            survivor_sel=SurvivorSelection("KeepBest"),
            params={"pcross": self.pcross, "pmut": self.pmut},
        )

        algorithm = GeneralAlgorithm(self.objfunc, strategy, params=self.genetic_params)

        best_solution, best_fitness = algorithm.optimize()
        self.medioids = X[best_solution, :]
        return self

    def predict(self, X):
        dist_mat = self.objfunc.compute_distance(X, self.medioids)
        return np.argmin(dist_mat, axis=1)
