import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.datasets import make_blobs, make_classification
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score, mean_squared_error

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
    def __init__(self, equation_str, X_train=None, y_train=None, metric_fn=None):
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
        self.metric_fn = metric_fn

        super().__init__(len(self.curve_params), mode="max", low_lim=-100, up_lim=100, name="Symbolic regression")

    def set_data(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def objective(self, vector):
        return self.metric_fn(self.y_train, self.predict(vector))


class ParametricSymbolicClassificationObj(ParametricSymbolicModelObj):
    def __init__(self, equation_str, X_train=None, y_train=None, metric_fn=None):
        if metric_fn is None:
            metric_fn = accuracy_score

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

class ParametricSymbolicRegressionObj(ParametricSymbolicModelObj):
    def __init__(self, equation_str, X_train=None, y_train=None, metric_fn=None):
        if metric_fn is None:
            metric_fn = mean_squared_error
    
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
