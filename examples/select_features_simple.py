from evolml.feature_selection import select_features, EvalFeatureSelectionMaskCV
from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from metaheuristic_designer.simple import *


def main(n_features, n_informative):
    X, y = make_regression(n_samples=500, n_features=n_features, n_informative=n_informative, noise=0.01, n_targets=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    baseline_model = Ridge()

    optim_alg = genetic_algorithm(
        objfunc=None, params={"stop_cond": "time_limit", "time_limit": 10.0, "min": 0, "max": n_features, "encoding": "int"}
    )

    mask, sparse_mask, score = select_features(optim_alg, baseline_model, X_train, y_train, n_informative - 1, cv_splits=5, cv_repeats=10, random_state=None)

    print(mask)
    print(sparse_mask)
    print(score)


if __name__ == "__main__":
    main(15, 7)
