"""
PyML Dataset Balancing Module
Author: Vaibhav Arun Patil
Version: 0.1.0
"""

import numpy as np
import pandas as pd
from ..core.base import BaseModule
from ..logger import logger
from ..exceptions import ConfigurationError

# Try importing SMOTE safely
try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False


class BalancingHandler(BaseModule):
    """
    Handles imbalanced dataset.
    """

    def __init__(self):
        super().__init__()
        logger.info("BalancingHandler Initialized")

    def balance(self, X, y, method="smote"):
        """
        Balance dataset.
        method: smote | none
        """

        if method == "smote":

            if not IMBLEARN_AVAILABLE:
                raise ConfigurationError(
                    "imblearn is required for SMOTE. Install using: pip install imbalanced-learn"
                )

            smote = SMOTE(random_state=self.config.RANDOM_STATE)
            X_res, y_res = smote.fit_resample(X, y)

            logger.info("SMOTE applied successfully.")
            return X_res, y_res

        elif method == "none":
            logger.info("No balancing applied.")
            return X, y

        else:
            raise ConfigurationError("Invalid balancing method.")