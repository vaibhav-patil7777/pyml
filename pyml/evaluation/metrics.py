"""
PyML Evaluation Metrics Module
Author: Vaibhav Arun Patil
Version: 0.1.0
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score
)

from ..logger import logger


class Metrics:

    def __init__(self):
        logger.info("Evaluation Metrics Module Initialized")

    # ================= CLASSIFICATION =================

    def classification_report(self, y_true, y_pred):

        report = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted"),
            "recall": recall_score(y_true, y_pred, average="weighted"),
            "f1_score": f1_score(y_true, y_pred, average="weighted"),
            "confusion_matrix": confusion_matrix(y_true, y_pred)
        }

        logger.info("Classification metrics calculated.")
        return report

    # ================= REGRESSION =================

    def regression_report(self, y_true, y_pred):

        report = {
            "mae": mean_absolute_error(y_true, y_pred),
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "r2_score": r2_score(y_true, y_pred)
        }

        logger.info("Regression metrics calculated.")
        return report

    # ================= CLUSTERING =================

    def clustering_report(self, X, labels):

        score = silhouette_score(X, labels)

        logger.info("Clustering metrics calculated.")
        return {
            "silhouette_score": score
        }