"""
PyML Auto Model Selection Engine
Author: Vaibhav Arun Patil
Version: 0.1.0
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from ..model.model_engine import ModelEngine
from ..logger import logger
from ..core.base import BaseModule


class AutoModelSelector(BaseModule):

    def __init__(self):
        super().__init__()
        self.model_engine = ModelEngine()
        logger.info("Auto Model Selector Initialized")

    def compare_models(self, X, y):

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE
        )

        model_names = [
            "linear_regression",
            "ridge",
            "lasso",
            "logistic_regression",
            "decision_tree_classifier",
            "random_forest_classifier",
            "random_forest_regressor",
            "kmeans"
        ]

        results = []

        for name in model_names:

            try:
                model = self.model_engine.get_model(name)

                if "cluster" in name or name == "kmeans":
                    model.fit(X_train)
                    labels = model.predict(X_test)
                    score = accuracy_score(y_test, labels)

                else:
                    model.fit(X_train, y_train)

                    if len(np.unique(y)) <= 20:
                        predictions = model.predict(X_test)
                        score = accuracy_score(y_test, predictions)
                    else:
                        predictions = model.predict(X_test)
                        score = r2_score(y_test, predictions)

                results.append({
                    "model": name,
                    "score": score
                })

            except:
                continue

        # Sort by best score
        results = sorted(results, key=lambda x: x["score"], reverse=True)

        logger.info("Model Comparison Completed.")

        return results

    def best_model(self, X, y):

        results = self.compare_models(X, y)

        if len(results) == 0:
            return None

        return results[0]