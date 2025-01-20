from copy import copy, deepcopy
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score
from metaheuristic_designer import ObjectiveFunc, ObjectiveVectorFunc, Encoding
from metaheuristic_designer.initializers import UniformVectorInitializer
from functools import reduce
import numpy as np

from ..utils import restart_model, evaluate_model


class SparseMaskEncoding(Encoding):
    def __init__(self, shape):
        self.shape = shape
        self.size = reduce(lambda x, y: x * y, shape)

    def encode(self, phenotype):
        flat_mat = phenotype.reshape((phenotype.shape[0], -1))
        pos_list = []
        for p in flat_mat:
            pos, *_ = np.where(p != 0)
            pos_list.append(pos)
        return pos_list

    def decode(self, genotype):
        if not isinstance(genotype.dtype, np.integer):
            genotype = genotype.astype(int)

        flat_mask = np.zeros((genotype.shape[0], self.size))
        flat_mask[np.arange(genotype.shape[0])[:, None], genotype] = 1
        return flat_mask


class EvalFeatureSelectionMaskCV(ObjectiveVectorFunc):
    def __init__(
        self,
        baseline_model,
        X_train,
        y_train,
        n_features,
        cross_validator=None,
        metric_fn=None,
        random_state=None,
        recalculate=False,
    ):
        self.baseline_model = baseline_model
        self.X_train = X_train
        self.y_train = y_train

        if cross_validator is None:
            cross_validator = RepeatedKFold(n_splits=5, n_repeats=10)

        if hasattr(cross_validator, "random_state"):
            cross_validator.random_state = random_state

        self.cross_validator = cross_validator

        if metric_fn is None:
            metric_fn = r2_score
        self.metric_fn = metric_fn

        self.random_state = random_state
        n_base_features = reduce(lambda x, y: x * y, X_train.shape[1:])

        super().__init__(
            n_features,
            mode="max",
            low_lim=0,
            up_lim=n_base_features - 1,
            recalculate=False,
            name="Evaluate Masked Feature Selection",
        )

    def objective(self, solution):
        X_train_masked = self.X_train[:, solution != 0]

        return evaluate_model(self.baseline_model, X_train_masked, self.y_train, metric_fn=self.metric_fn, cross_validator=self.cross_validator)

    def repair_solution(self, vector):
        clipped_vector = super().repair_solution(vector)
        _, inverse, counts = np.unique(clipped_vector, return_inverse=True, return_counts=True)
        duplicate_indices = counts[inverse] > 1

        if not np.any(duplicate_indices):
            return clipped_vector

        choices = np.setdiff1d(np.arange(self.X_train.shape[1]), clipped_vector)
        new_values = np.random.choice(choices, np.count_nonzero(duplicate_indices), replace=False)
        clipped_vector[duplicate_indices] = new_values
        return clipped_vector


def select_features(
    optim_algorithm,
    baseline_model,
    X_train,
    y_train,
    n_features,
    cv_splits=5,
    cv_repeats=10,
    random_state=None,
    pop_size=100,
):
    baseline_model = restart_model(baseline_model)

    if optim_algorithm.initializer is not None:
        pop_size = optim_algorithm.initializer.pop_size

    cv = RepeatedKFold(n_splits=cv_splits, n_repeats=cv_repeats)
    objfunc = EvalFeatureSelectionMaskCV(baseline_model, X_train, y_train, n_features, cross_validator=cv, random_state=random_state)
    encoding = SparseMaskEncoding(X_train.shape[1:])
    initializer = UniformVectorInitializer(
        objfunc.vecsize,
        objfunc.low_lim,
        objfunc.up_lim,
        pop_size=pop_size,
        encoding=encoding,
        dtype=int,
    )
    optim_algorithm.objfunc = objfunc
    optim_algorithm.initializer = initializer
    result = optim_algorithm.optimize()
    best_solution, best_fitness = result.best_solution()
    best_solution_mask, _ = result.best_solution(decoded=True)
    return best_solution_mask, best_solution, best_fitness
