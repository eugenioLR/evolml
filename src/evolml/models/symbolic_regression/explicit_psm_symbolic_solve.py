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

from .symbolic_model_objective import ParametricSymbolicClassificationObj, ParametricSymbolicRegressionObj

class ExplicitPSRegressorSymbSolve(BaseEstimator, RegressorMixin):
    """
    PS (Parametrized symbolic) classifier.
    """

    def __init__(self, expression, optim_params=None):
        if optim_params is None:
            optim_params = {
                "stop_cond": "time_limit or convergence",
                "time_limit": 60.0,
                "cpu_time_limit": 10.0,
                "ngen": 100,
                "neval": 6e5,
                "fit_target": 1.0,
                "patience": 100,
                "verbose": True,
                "v_timer": 0.5,
            }
        self.optim_params = optim_params

        self.expression = expression
        self.objfunc = ParametricSymbolicRegressionObj(expression)
        # self.initializer = initializers.UniformVectorInitializer(self.objfunc.vecsize, self.objfunc.low_lim, self.objfunc.up_lim, pop_size=100)
        # self.search_strat = strategies.PSO(self.initializer, {"w": 0.7, "c1": 1.5, "c2": 1.5})
        # self.optim_algorithm = algorithms.GeneralAlgorithm(self.objfunc, self.search_strat, params=optim_params)
        self.parameters = None

    def fit(self, X, y):
        # self.objfunc.set_data(X, y)
        # self.parameters, fit = self.optim_algorithm.optimize()
        # self.model_eq = self.objfunc.equation.subs(zip(self.objfunc.curve_params, self.parameters))
        # self.model_fn = sympy.lambdify(self.objfunc.input_params, self.model_eq)
        # symbolic_mse = sum((self.objfunc.equation.subs()))

        equation = self.objfunc.equation
        input_params = self.objfunc.input_params
        curve_params = self.objfunc.curve_params
        symbolic_mse = 0
        for x_i, y_i in zip(X, y):
            print(equation.subs(zip(input_params, x_i)))
            print(type(equation.subs(zip(input_params, x_i))))
            symbolic_mse += (equation.subs(zip(input_params, x_i)) - float(y_i))**2
        
        # symbolic_mse = sympy.simplify(symbolic_mse)
        print(symbolic_mse)
        print(type(symbolic_mse))
        # grad_mse = [sympy.diff(symbolic_mse, param) for param in curve_params]
        # gradient = lambda f, v: sympy.Matrix([f]).jacobian(v)
        # grad_mse = gradient(symbolic_mse, curve_params)
        grad_mse = []
        for param in curve_params:
            print(param)
            partial = sympy.diff(symbolic_mse, param)
            grad_mse.append(partial)
        
        print(grad_mse)
        # regression_function = sympy.Eq(grad_mse, sympy.symbols('y'))
        # print(regression_function)

        optimal_params = sympy.solve(grad_mse, curve_params)
        print(optimal_params)

        min_values = [(equation.subs(zip(input_params, x_min)), x_min) for x_min in optimal_params]

        _, self.parameters = min(min_values, key=lambda item: item[0])

        return self

    def predict(self, X):
        pred = np.empty((X.shape[0], self.objfunc.y_train.shape[1]))
        for idx, data_point in enumerate(X):
            pred[idx] = float(self.model_fn(*data_point))
        return pred
