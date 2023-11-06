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
from metaheuristic_designer import ObjectiveFunc, ObjectiveVectorFunc
from metaheuristic_designer.simple import *


class ParametricSymbolicModelObj(ObjectiveVectorFunc):
    def __init__(self, equation_str, X_train=None, y_train=None):
        self.equation_str = equation_str
        self.equation = sympy.parsing.sympy_parser.parse_expr(equation_str)
        self.equation = sympy.simplify(self.equation)
        atoms = self.equation.atoms()

        curve_params = list()
        input_params = list()
        for i in atoms:
            if type(i) is sympy.Symbol:
                if str(i)[0] == "p" and i not in curve_params:
                    curve_params.append(i)
                elif str(i)[0] == "x" and i not in input_params:
                    input_params.append(i)
        self.curve_params = sorted(curve_params, key=lambda x: int(str(x).split("_")[-1]))
        self.input_params = sorted(input_params, key=lambda x: int(str(x).split("_")[-1]))
        self.X_train = X_train
        self.y_train = y_train

        super().__init__(len(self.curve_params), mode="max", low_lim=-100, up_lim=100, name="Symbolic regression")

    def set_data(self, X, y):
        self.X_train = X
        self.y_train = y


class ParametricSymbolicClassificationObj(ParametricSymbolicModelObj):
    def decision_boundary(self):
        return sympy.Eq(self.equation, 0)

    def predict(self, vector):
        try:
            model_eq = self.equation.subs(zip(self.curve_params, vector))
            model = sympy.lambdify(self.input_params, model_eq)

            pred = np.empty_like(self.y_train)
            for idx, data_point in enumerate(self.X_train):
                # pred[idx] = int(model_eq.subs(zip(self.input_params, data_point)) > 0)
                pred[idx] = int(model(*data_point) > 0)

        except Exception as e:
            print(e)
            pred = np.zeros_like(self.y_train)

        return pred

    def objective(self, vector):
        return roc_auc_score(self.y_train, self.predict(vector))


class ParametricSymbolicRegressionObj(ParametricSymbolicModelObj):
    def decision_boundary(self):
        return sympy.Eq(self.equation, sympy.symbols(f"x_{len(self.input_params)}"))

    def predict(self, vector):
        try:
            model_eq = self.equation.subs(zip(self.curve_params, vector))
            model = sympy.lambdify(self.input_params, model_eq)

            pred = np.empty_like(self.y_train)
            for idx, data_point in enumerate(self.X_train):
                pred[idx] = float(model(*data_point))

        except Exception as e:
            print(e)
            pred = np.zeros_like(self.y_train)

        return pred

    def objective(self, vector):
        return r2_score(self.y_train, self.predict(vector))


class ExplicitPSClassifier(BaseEstimator, ClassifierMixin):
    """
    PS (Parametrized symbolic) classifier.
    """

    def __init__(self, expression, params=None):
        if params is None:
            params = {
                "stop_cond": "time_limit or convergence or fit_target",
                "time_limit": 60.0,
                "cpu_time_limit": 10.0,
                "ngen": 100,
                "neval": 6e5,
                "fit_target": 1.0,
                "patience": 15,
                "verbose": True,
                "v_timer": 0.5,
            }

        self.objfunc = ParametricSymbolicClassificationObj(expression)
        self.initializer = initializers.UniformVectorInitializer(self.objfunc.vecsize, self.objfunc.low_lim, self.objfunc.up_lim, pop_size=100)
        self.search_strat = strategies.PSO(self.initializer, {"w": 0.7, "c1": 1.5, "c2": 1.5})
        self.search_algorithm = algorithms.GeneralAlgorithm(self.objfunc, self.search_strat, params=params)
        self.parameters = None

    def fit(self, X, y):
        self.objfunc.set_data(X, y)
        self.parameters, fit = self.search_algorithm.optimize()
        self.model_eq = self.objfunc.equation.subs(zip(self.objfunc.curve_params, self.parameters))
        self.model_fn = sympy.lambdify(self.objfunc.input_params, self.model_eq)
        return self

    def predict(self, X):
        pred = np.empty((X.shape[0], self.objfunc.y_train.shape[1]))
        for idx, data_point in enumerate(X):
            pred[idx] = int(self.model_fn(*data_point) > 0)
        return pred


class ExplicitPSRegressor(BaseEstimator, RegressorMixin):
    """
    PS (Parametrized symbolic) classifier.
    """

    def __init__(self, expression, params=None):
        if params is None:
            params = {
                "stop_cond": "time_limit or convergence or fit_target",
                "time_limit": 60.0,
                "cpu_time_limit": 10.0,
                "ngen": 100,
                "neval": 6e5,
                "fit_target": 1.0,
                "patience": 15,
                "verbose": True,
                "v_timer": 0.5,
            }

        self.objfunc = ParametricSymbolicRegressionObj(expression)
        self.initializer = initializers.UniformVectorInitializer(self.objfunc.vecsize, self.objfunc.low_lim, self.objfunc.up_lim, pop_size=100)
        self.search_strat = strategies.PSO(self.initializer, {"w": 0.7, "c1": 1.5, "c2": 1.5})
        self.search_algorithm = algorithms.GeneralAlgorithm(self.objfunc, self.search_strat, params=params)
        self.parameters = None

    def fit(self, X, y):
        self.objfunc.set_data(X, y)
        self.parameters, fit = self.search_algorithm.optimize()
        self.model_eq = self.objfunc.equation.subs(zip(self.objfunc.curve_params, self.parameters))
        self.model_fn = sympy.lambdify(self.objfunc.input_params, self.model_eq)
        return self

    def predict(self, X):
        pred = np.empty((X.shape[0], self.objfunc.y_train.shape[1]))
        for idx, data_point in enumerate(X):
            pred[idx] = float(self.model_fn(*data_point))
        return pred
