from copy import copy
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold


def evaluate_model(model_base, X_train, y_train, metric_fn=None, cross_validator=None):
    if metric_fn is None:
        metric_fn = mean_squared_error
    
    if cross_validator is None:
        cross_validator = RepeatedKFold(n_splits=5, n_repeats=10)

    final_score = 0
    for i, (train_index, test_index) in enumerate(cross_validator.split(X_train)):
        X_train_cv = X_train[train_index]
        y_train_cv = y_train[train_index]
        X_test_cv = X_train[test_index]
        y_test_cv = y_train[test_index]

        model = copy(model_base).fit(X_train_cv, y_train_cv)
        y_pred = model.predict(X_test_cv)
        final_score += metric_fn(y_pred, y_test_cv)


    return final_score / cross_validator.get_n_splits()
