from sklearn.base import MultiOutputMixin
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputClassifier
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
        flat_mat = phenotype.flatten()
        pos, *_ = np.where(flat_mat != 0)
        return pos

    def decode(self, genotype):
        if not isinstance(genotype.dtype, np.integer):
            genotype = genotype.astype(int)

        flat_mask = np.zeros(self.size)
        flat_mask[[genotype]] = 1
        return flat_mask.reshape(self.shape)


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
            name="Evaluate Masked Feature Selection",
        )

    def objective(self, vector):
        X_train_masked = self.X_train[:, vector != 0]

        return evaluate_model(self.baseline_model, X_train_masked, self.y_train, metric_fn=self.metric_fn, cross_validator=self.cross_validator)


class EvalFeatureSelectionMaskMultiOutputCV(ObjectiveVectorFunc):
    def __init__(
        self,
        baseline_model,
        X_train,
        y_train,
        n_features,
        target_weights=None,
        cross_validator=None,
        metric_fn=None,
        random_state=None,
    ):
        if not isinstance(baseline_model, MultiOutputMixin):
            baseline_model = MultiOutputClassifier(baseline_model)

        self.baseline_model = baseline_model
        self.X_train = X_train
        self.y_train = y_train

        if target_weights is None:
            target_weights = np.full(1/y_train.shape[1], y_train.shape[1])
        assert target_weights.shape[0] == y_train.shape[1] 
        self.target_weights = target_weights

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
            name="Evaluate Masked Feature Selection",
        )

    def objective(self, vector):
        X_train_masked = self.X_train[:, vector != 0]

        metrics = np.empty(self.y_train.shape[1])
        for idx, target_i in enumerate(self.y_train.T):
            metrics[idx] = evaluate_model(self.baseline_model, X_train_masked, target_i, metric_fn=self.metric_fn, cross_validator=self.cross_validator)
        
        return np.average(metrics, weights=self.class_weights)


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
        pop_size=100,
        encoding=encoding,
        dtype=int,
    )
    optim_algorithm.objfunc = objfunc
    optim_algorithm.initializer = initializer
    best_solution, best_fitness = optim_algorithm.optimize()
    return encoding.decode(best_solution), best_solution, best_fitness
