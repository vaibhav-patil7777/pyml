"""
PyML Scaling Module
Author: Vaibhav Arun Patil
Version: 0.1.0
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from ..core.base import BaseModule
from ..logger import logger
from ..exceptions import ConfigurationError


class ScalingHandler(BaseModule):
    """
    Handles feature scaling.
    """

    def __init__(self):
        super().__init__()
        self.scalers = {}
        logger.info("ScalingHandler Initialized")

    def scale(self, data, method="standard", columns=None):
        """
        Apply feature scaling.
        method: standard | minmax | robust
        """

        data = self.convert_to_dataframe(data)

        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns

        if method == "standard":
            scaler = StandardScaler()

        elif method == "minmax":
            scaler = MinMaxScaler()

        elif method == "robust":
            scaler = RobustScaler()

        else:
            raise ConfigurationError("Invalid scaling method.")

        data[columns] = scaler.fit_transform(data[columns])
        self.scalers[method] = scaler

        logger.info(f"{method} scaling applied.")
        return data

    def get_scaler(self, method):
        """
        Return trained scaler object.
        """
        return self.scalers.get(method, None)