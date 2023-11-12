from copy import copy
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score
from metaheuristic_designer import ObjectiveFunc, ObjectiveVectorFunc, Encoding
from itertools import reduce 

class SparseMaskEncoding(Encoding):
    def __init__(self, shape):
        self.shape = shape
        self.size = reduce(lambda x, y: x*y, shape)

    def encode(self, phenotype):
        flat_mat = phenotype.flatten()
        pos, *_ = np.where(flat_mat != 0)
        return pos        

    def decode(self, genotype):
        flat_mask = np.zeros(size)
        flat_mask[genotype] = 1
        return flat_mask.resize(self.shape)


class EvalFeatureSelectionCV(ObjectiveVectorFunc):
    def __init__(self, baseline_model, X_train, y_train, n_features, cv_splits=5, cv_repeats=10, random_state=None):
        self.baseline_model = baseline_model
        self.X_train = X_train
        self.y_train = y_train
        self.cv_splits = cv_splits
        self.cv_repeats = cv_repeats
        self.random_state = random_state

        super().__init__(n_features, mode="max", low_lim=0, up_lim=1, name="Evaluate Feature Selection")
    
    def objective(self, vector):
        X_train_masked = self.X_test[:, vector != 0]

        cv_eval = RepeatedKFold(n_splits=self.cv_splits, n_repeats=self.cv_repeats, random_state=random_state)

        n_evals = 0
        final_score = 0
        for i, (train_index, test_index) in enumerate(cv_eval.split(X_train_masked)):
            X_train_cv = X_train_masked[train_index, :]
            y_train_cv = y_train_masked[train_index, :]
            X_test_cv = X_train_masked[test_index, :]
            y_test_cv = y_train_masked[test_index, :]

            model = copy(self.baseline_model).fit(X_train_cv, y_train_cv)
            y_pred = model.predict(X_test_cv)
            final_score += r2_score(y_pred, y_test_cv)

            n_evals += 1
        
        return final_score/n_evals


def select_features(baseline_model, X_train, y_train):
    pass