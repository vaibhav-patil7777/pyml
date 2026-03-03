"""
PyML Missing Value Handling Module
Author: Vaibhav Arun Patil
Version: 0.1.0
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from ..core.base import BaseModule
from ..logger import logger
from ..exceptions import MissingValueError


class MissingValueHandler(BaseModule):
    """
    Handles missing values for numerical and categorical data.
    Sklearn-compatible design.
    """

    def __init__(self):
        super().__init__()
        self.num_imputer = None
        self.cat_imputer = None
        self.numeric_cols = None
        self.categorical_cols = None
        logger.info("MissingValueHandler Initialized")

    # ===============================
    # FIT METHOD
    # ===============================
    def fit(self, data, y=None):
        """
        Learn imputation strategies from data.
        """
        data = self.convert_to_dataframe(data)
        self.validate_data(data)

        self.numeric_cols = data.select_dtypes(include=[np.number]).columns
        self.categorical_cols = data.select_dtypes(exclude=[np.number]).columns

        # Fit Numerical Imputer
        if len(self.numeric_cols) > 0:
            self.num_imputer = SimpleImputer(strategy="mean")
            self.num_imputer.fit(data[self.numeric_cols])

        # Fit Categorical Imputer
        if len(self.categorical_cols) > 0:
            self.cat_imputer = SimpleImputer(strategy="most_frequent")
            self.cat_imputer.fit(data[self.categorical_cols])

        return self

    # ===============================
    # TRANSFORM METHOD
    # ===============================
    def transform(self, data):
        """
        Apply learned imputation to data.
        """
        data = self.convert_to_dataframe(data)

        # Transform Numerical Columns
        if self.num_imputer is not None and len(self.numeric_cols) > 0:
            data[self.numeric_cols] = self.num_imputer.transform(
                data[self.numeric_cols]
            )
            logger.info("Numerical missing values handled.")

        # Transform Categorical Columns
        if self.cat_imputer is not None and len(self.categorical_cols) > 0:
            data[self.categorical_cols] = self.cat_imputer.transform(
                data[self.categorical_cols]
            )
            logger.info("Categorical missing values handled.")

        if data.isnull().sum().sum() != 0:
            raise MissingValueError("Some missing values could not be handled.")

        return data

    # ===============================
    # FIT_TRANSFORM METHOD
    # ===============================
    def fit_transform(self, data, y=None):
        """
        Sklearn-style fit and transform.
        """
        return self.fit(data).transform(data)

    # ===============================
    # MISSING REPORT
    # ===============================
    def get_missing_report(self, data):
        """
        Return missing value percentage report.
        """
        data = self.convert_to_dataframe(data)
        report = (data.isnull().sum() / len(data)) * 100
        return report