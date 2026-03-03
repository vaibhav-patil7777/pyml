"""
PyML Universal Model Engine
Author: Vaibhav Arun Patil
Version: 1.0.0
"""

import numpy as np

# Regression
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    BayesianRidge, SGDRegressor
)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor
)
from sklearn.neighbors import KNeighborsRegressor

# Classification
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# Clustering
from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering,
    MeanShift, SpectralClustering
)

from pyml.core.base import BaseModule
from pyml.logger import logger


class ModelEngine(BaseModule):

    def __init__(self):
        super().__init__()
        logger.info("ModelEngine initialized")

    def get_model(self, model_name):

        rs = self.config.RANDOM_STATE

        models = {

            # Regression
            "linear_regression": LinearRegression(),
            "ridge": Ridge(),
            "lasso": Lasso(),
            "elasticnet": ElasticNet(),
            "bayesian_ridge": BayesianRidge(),
            "sgd_regressor": SGDRegressor(),
            "svr": SVR(),
            "decision_tree_regressor": DecisionTreeRegressor(),
            "random_forest_regressor": RandomForestRegressor(random_state=rs),
            "gradient_boosting_regressor": GradientBoostingRegressor(),
            "extra_trees_regressor": ExtraTreesRegressor(),
            "knn_regressor": KNeighborsRegressor(),

            # Classification
            "logistic_regression": LogisticRegression(max_iter=2000),
            "sgd_classifier": SGDClassifier(),
            "svc": SVC(),
            "decision_tree_classifier": DecisionTreeClassifier(),
            "random_forest_classifier": RandomForestClassifier(random_state=rs),
            "gradient_boosting_classifier": GradientBoostingClassifier(),
            "extra_trees_classifier": ExtraTreesClassifier(),
            "knn_classifier": KNeighborsClassifier(),
            "gaussian_nb": GaussianNB(),
            "multinomial_nb": MultinomialNB(),
            "bernoulli_nb": BernoulliNB(),

            # Clustering
            "kmeans": KMeans(random_state=rs),
            "dbscan": DBSCAN(),
            "agglomerative": AgglomerativeClustering(),
            "meanshift": MeanShift(),
            "spectral": SpectralClustering()
        }

        if model_name not in models:
            raise ValueError(f"Model '{model_name}' not supported.")

        return models[model_name]

    def train(self, X, y=None, model_name="auto"):

        if model_name == "auto":
            if y is None:
                model_name = "kmeans"
            elif len(np.unique(y)) <= 10:
                model_name = "random_forest_classifier"
            else:
                model_name = "random_forest_regressor"

        model = self.get_model(model_name)

        if y is not None:
            model.fit(X, y)
        else:
            model.fit(X)

        logger.info(f"{model_name} trained successfully")
        return model