from __future__ import annotations
from typing import Tuple
import torch
import scipy as sp
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.cluster import KMeans
from .rbf_model import RBFNNModel


class RBFNNClassifier(RBFNNModel):
    """ """

    def __init__(
        self,
        n_units: int,
        probability: float = True,
        linear_layer: BaseEstimator = None,
        cluster_model: BaseEstimator = None,
        std_from_clusters: bool = False,
        random_state: int = None,
    ):
        self.probability = probability

        if linear_layer is None:
            if probability:
                linear_layer = LogisticRegression(penalty=None, random_state=random_state)
            else:
                linear_layer = RidgeClassifier(alpha=0, solver="svd", random_state=random_state)
        
        super().__init__(
            n_units=n_units,
            linear_layer=linear_layer,
            cluster_model=cluster_model,
            std_from_clusters=std_from_clusters,
            classification=True,
            random_state=random_state,
        )
