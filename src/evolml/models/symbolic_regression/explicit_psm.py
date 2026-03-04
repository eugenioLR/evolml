from __future__ import annotations
from abc import ABC, abstractmethod
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.datasets import make_blobs, make_classification
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("dark")

import sympy
import sympy.abc
from sympy.plotting.plot import MatplotlibBackend, Plot

import metaheuristic_designer as pec
from metaheuristic_designer import algorithms
from metaheuristic_designer import strategies
from metaheuristic_designer import operators
from metaheuristic_designer import initializers
from metaheuristic_designer import ObjectiveFunc, VectorObjectiveFunc
from metaheuristic_designer.simple import *

from .symbolic_model_objective import ParametricSymbolicClassificationObj, ParametricSymbolicRegressionObj


class ExplicitPSModel(ABC, BaseEstimator, ClassifierMixin):
    """
    PS (Parametrized symbolic) classifier.
    """

    def __init__(self, expression, objfunc, optim_algorithm=None):
        self.expression = expression
        self.objfunc = objfunc
        initializer = initializers.UniformInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=100)
        if optim_algorithm is None:
            optim_params = {
                "stop_cond": "time_limit or convergence",
                "time_limit": 60.0,
                "cpu_time_limit": 10.0,
                "ngen": 100,
                "neval": 6e5,
                "fit_target": 1.0,
                "patience": 10,
                "verbose": True,
                "v_timer": 0.5,
            }
            search_strat = strategies.PSO(initializer, {"w": 0.7, "c1": 1.5, "c2": 1.5})
            optim_algorithm = algorithms.GeneralAlgorithm(objfunc, search_strat, params=optim_params)
        else:
            optim_algorithm.initializer = initializer
            optim_algorithm.objfunc = objfunc
        self.optim_algorithm = optim_algorithm
        self.parameters = None

    def fit(self, X, y):
        self.objfunc.set_data(X, y)
        result = self.optim_algorithm.optimize()
        self.parameters, fit = result.best_solution()
        self.model_eq = self.objfunc.equation.subs(zip(self.objfunc.curve_params, self.parameters))
        self.model_fn = sympy.lambdify(self.objfunc.input_params, self.model_eq)
        return self


class ExplicitPSClassifier(ExplicitPSModel):
    """
    PS (Parametrized symbolic) classifier.
    """

    def __init__(self, expression, optim_algorithm=None):
        super().__init__(expression=expression, objfunc=ParametricSymbolicClassificationObj(expression), optim_algorithm=optim_algorithm)

    def predict(self, X):
        pred = np.empty((X.shape[0], self.objfunc.y_train.shape[1]))
        for idx, data_point in enumerate(X):
            pred[idx] = int(self.model_fn(*data_point) > 0)
        return pred


class ExplicitPSRegressor(ExplicitPSModel):
    """
    PS (Parametrized symbolic) regressor.
    """

    def __init__(self, expression, optim_algorithm=None):
        super().__init__(expression=expression, objfunc=ParametricSymbolicRegressionObj(expression), optim_algorithm=optim_algorithm)

    def predict(self, X):
        pred = np.empty((X.shape[0], self.objfunc.y_train.shape[1]))
        for idx, data_point in enumerate(X):
            pred[idx] = float(self.model_fn(*data_point))
        return pred
