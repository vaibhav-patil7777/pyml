"""
PyML Model Explainability Module
Author: Vaibhav Arun Patil
Version: 0.1.0
"""

import numpy as np
import pandas as pd

from ..logger import logger


class ModelExplainer:

    def __init__(self):
        logger.info("Model Explainability Module Initialized")

    def feature_importance(self, model, feature_names):

        if not hasattr(model, "feature_importances_"):
            logger.info("Model does not support feature importance.")
            return None

        importances = model.feature_importances_

        result = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        })

        result = result.sort_values(by="importance", ascending=False)

        logger.info("Feature importance calculated.")

        return result

    def linear_model_coefficients(self, model, feature_names):

        if not hasattr(model, "coef_"):
            logger.info("Model does not support coefficients.")
            return None

        coefficients = model.coef_

        if len(coefficients.shape) > 1:
            coefficients = coefficients[0]

        result = pd.DataFrame({
            "feature": feature_names,
            "coefficient": coefficients
        })

        result = result.sort_values(by="coefficient", ascending=False)

        logger.info("Model coefficients extracted.")

        return result