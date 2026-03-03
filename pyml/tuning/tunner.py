"""
PyML Hyperparameter Tuning Module
Author: Vaibhav Arun Patil
Version: 0.1.0
"""

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from ..logger import logger
from ..exceptions import ModelTrainingError


class Tuner:

    def __init__(self):
        logger.info("Hyperparameter Tuner Initialized")

    def grid_search(self, model, param_grid, X, y, cv=5):
        """
        Perform Grid Search Cross Validation.
        """

        try:
            grid = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=cv,
                n_jobs=-1
            )

            grid.fit(X, y)

            logger.info("Grid Search Completed.")
            return grid.best_estimator_, grid.best_params_

        except Exception as e:
            raise ModelTrainingError(str(e))

    def random_search(self, model, param_distributions, X, y, cv=5, n_iter=10):
        """
        Perform Randomized Search Cross Validation.
        """

        try:
            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_distributions,
                n_iter=n_iter,
                cv=cv,
                n_jobs=-1
            )

            random_search.fit(X, y)

            logger.info("Random Search Completed.")
            return random_search.best_estimator_, random_search.best_params_

        except Exception as e:
            raise ModelTrainingError(str(e))