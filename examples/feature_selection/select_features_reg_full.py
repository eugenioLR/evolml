from evolml.feature_selection import SparseMaskEncoding, EvalFeatureSelectionMaskCV
from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from metaheuristic_designer.strategies import GA, SA, LocalSearch
from metaheuristic_designer.operators import OperatorInt
from metaheuristic_designer.algorithms import GeneralAlgorithm
from metaheuristic_designer.initializers import UniformVectorInitializer
from metaheuristic_designer.selectionMethods import ParentSelection, SurvivorSelection

def main(n_features, n_informative):
    X, y = make_regression(n_samples=1500, n_features=n_features, n_informative=n_informative, noise=0.01, n_targets=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    baseline_model = Ridge()
    
    objfunc = EvalFeatureSelectionMaskCV(baseline_model, X_train, y_train, n_informative, cv_splits=5, cv_repeats=10) 
    encoding = SparseMaskEncoding(X_train.shape[1:])
    initializer = UniformVectorInitializer(objfunc.vecsize, objfunc.low_lim, objfunc.up_lim, pop_size=80, encoding=encoding, dtype=int)

    mutate_op = OperatorInt("MutSample", {"distrib": "Uniform", "min": 0, "max": n_features, "N": 1})
    cross_op = OperatorInt("Multipoint")
    parent_sel = ParentSelection("Tournament", {"amount":50, "p": 0.05}) 
    surv_sel = SurvivorSelection("Elitism", {"amount": 5})

    search_strategy = GA(initializer, mutate_op, cross_op, parent_sel_op=parent_sel, selection_op=surv_sel, params={"pmut": 0.05, "pcross": 0.9})

    optim_algorithm = GeneralAlgorithm(objfunc, search_strategy, params={
        "stop_cond": "time_limit",
        "time_limit": 30.0,
        "verbose": True,
        "v_timer": 2
    })

    sparse_mask, score = optim_algorithm.optimize()
    mask = encoding.decode(sparse_mask)

    print(f"binary feature mask: {mask}")
    print(f"selected feature indexes: {sparse_mask}")
    print(f"R2 score in cross-validation: {score}")


if __name__ == "__main__":
    main(15, 7)
