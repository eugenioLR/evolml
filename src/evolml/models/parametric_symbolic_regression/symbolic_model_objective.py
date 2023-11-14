import sklearn
from sklearn.datasets import make_blobs, make_classification
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("dark")

import sympy
import sympy.abc
from sympy.plotting.plot import MatplotlibBackend, Plot

import pyevolcomp as pec
from pyevolcomp import Algorithms
from pyevolcomp import Operators
from pyevolcomp import SearchMethods
from pyevolcomp import ObjectiveFunc, ObjectiveVectorFunc
from pyevolcomp.simple import *


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