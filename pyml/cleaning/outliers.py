"""
PyML Outlier Handling Module
Author: Vaibhav Arun Patil
Version: 0.1.0
"""

import pandas as pd
import numpy as np
from scipy import stats
from ..core.base import BaseModule
from ..logger import logger


class OutlierHandler(BaseModule):
    """
    Handles outlier detection and removal.
    """

    def __init__(self):
        super().__init__()
        logger.info("OutlierHandler Initialized")

    def detect_iqr(self, data, column):
        """
        Detect outliers using IQR method.
        """
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = data[
            (data[column] < lower_bound) |
            (data[column] > upper_bound)
        ]

        return outliers

    def detect_zscore(self, data, column, threshold=3):
        """
        Detect outliers using Z-score method.
        """
        z_scores = np.abs(stats.zscore(data[column]))
        outliers = data[z_scores > threshold]
        return outliers

    def remove_outliers_iqr(self, data, column):
        """
        Remove outliers using IQR method.
        """
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        cleaned_data = data[
            (data[column] >= lower_bound) &
            (data[column] <= upper_bound)
        ]

        logger.info(f"Outliers removed using IQR from {column}")
        return cleaned_data

    def remove_outliers_zscore(self, data, column, threshold=3):
        """
        Remove outliers using Z-score method.
        """
        z_scores = np.abs(stats.zscore(data[column]))
        cleaned_data = data[z_scores <= threshold]

        logger.info(f"Outliers removed using Z-score from {column}")
        return cleaned_data