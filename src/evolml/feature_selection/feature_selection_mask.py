from copy import copy, deepcopy
from ..utils import restart_model
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score
from metaheuristic_designer import ObjectiveFunc, ObjectiveVectorFunc, Encoding
from metaheuristic_designer.initializers import UniformVectorInitializer
from functools import reduce
import numpy as np


class SparseMaskEncoding(Encoding):
    def __init__(self, shape):
        self.shape = shape
        self.size = reduce(lambda x, y: x * y, shape)

    def encode(self, phenotype):
        flat_mat = phenotype.flatten()
        pos, *_ = np.where(flat_mat != 0)
        return pos

    def decode(self, genotype):
        flat_mask = np.zeros(self.size)
        flat_mask[[genotype]] = 1
        return flat_mask.reshape(self.shape)


class EvalFeatureSelectionMaskCV(ObjectiveVectorFunc):
    def __init__(self, baseline_model, X_train, y_train, n_features, cv_splits=5, cv_repeats=10, metric_fn=None, random_state=None):
        self.baseline_model = baseline_model
        self.X_train = X_train
        self.y_train = y_train
        self.cv_splits = cv_splits
        self.cv_repeats = cv_repeats
        self.random_state = random_state
        if metric_fn is None:
            metric_fn = r2_score
        self.metric_fn = metric_fn
        n_base_features = reduce(lambda x, y: x * y, X_train.shape[1:])

        super().__init__(n_features, mode="max", low_lim=0, up_lim=n_base_features - 1, name="Evaluate Masked Feature Selection")

    def objective(self, vector):
        X_train_masked = self.X_train[:, vector != 0]

        cv_eval = RepeatedKFold(n_splits=self.cv_splits, n_repeats=self.cv_repeats, random_state=self.random_state)

        n_evals = 0
        final_score = 0
        for i, (train_index, test_index) in enumerate(cv_eval.split(X_train_masked)):
            X_train_cv = X_train_masked[train_index]
            y_train_cv = self.y_train[train_index]
            X_test_cv = X_train_masked[test_index]
            y_test_cv = self.y_train[test_index]

            model = copy(self.baseline_model).fit(X_train_cv, y_train_cv)
            y_pred = model.predict(X_test_cv)
            final_score += self.metric_fn(y_pred, y_test_cv)

            n_evals += 1

        return final_score / n_evals


def select_features(optim_algorithm, baseline_model, X_train, y_train, n_features, cv_splits=5, cv_repeats=10, random_state=None, pop_size=100):
    baseline_model = restart_model(baseline_model)

    if optim_algorithm.initializer is not None:
        pop_size = optim_algorithm.initializer.pop_size

    objfunc = EvalFeatureSelectionMaskCV(baseline_model, X_train, y_train, n_features, cv_splits, cv_repeats, random_state)
    encoding = SparseMaskEncoding(X_train.shape[1:])
    initializer = UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=100, encoding=encoding, dtype=int)
    optim_algorithm.objfunc = objfunc
    optim_algorithm.initializer = initializer
    best_solution, best_fitness = optim_algorithm.optimize()
    return encoding.decode(best_solution), best_solution, best_fitness
