from __future__ import annotations
from typing import Tuple
from abc import ABC, abstractmethod
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.base import BaseEstimator
# from .config import hh_logger
# from .hyperparameters import find_hyperparam_grid, find_hyperparam_random
# from .metric_alias import process_metric
from hundred_hammers import HyperOptimizer

class HyperOptimizerGA(HyperOptimizer):
    """
    Grid Search Hyperparameter Optimizer.

    :param metric: function that calculates the error of the predictions of a model compared with the real dataset.
    :type metric: str or callable or Tuple[str, callable, dict]
    :param n_folds_tune: number of splits in cross validation for grid search.
    :type n_folds_tune: int
    :param n_iter: amount of samples to take for each parameter.
    :type n_iter: int
    """

    def __init__(self, metric: str | callable = "MSE", idk=0):
        super().__init__(metric)
        # self.n_folds_tune = n_folds_tune
        # self.n_iter = n_iter

    def best_params(self, X: np.ndarray, y: np.ndarray, model: BaseEstimator, param_grid: dict = None):
        return NotImplementedError()

        if not param_grid:
            hh_logger.info(f"No specified hyperparameters for {type(model).__name__}. Generating hyperparameter distributions.")
            param_def = find_hyperparam_def(model)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid_search_model = RandomizedSearchCV(model, param_grid, scoring=self.metric_fn, n_jobs=-1, cv=self.n_folds_tune, n_iter=self.n_iter)
            grid_search_model.fit(X, y)

        results = pd.DataFrame(grid_search_model.cv_results_).dropna()
        best_params_df = results[results["rank_test_score"] == results["rank_test_score"].min()]
        best_params = best_params_df.head(1)["params"].values[0]

        return best_params
