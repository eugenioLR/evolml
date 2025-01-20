import torch
import scipy as sp
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.cluster import KMeans
from .rbf_model import RBFNNModel


class RBFNNRegressor(RBFNNModel):
    """ """

    def __init__(
        self,
        n_units: int,
        linear_layer: BaseEstimator = None,
        cluster_model: BaseEstimator = None,
        std_from_clusters: bool = False,
        random_state: int = None,
    ):
        if linear_layer is None:
            linear_layer = LinearRegression()

        super().__init__(
            n_units=n_units,
            linear_layer=linear_layer,
            cluster_model=cluster_model,
            std_from_clusters=std_from_clusters,
            classification=False,
            random_state=random_state,
        )
