"""
PyML Encoding Module
Author: Vaibhav Arun Patil
Version: 0.1.0
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from ..core.base import BaseModule
from ..logger import logger


class EncodingHandler(BaseModule):
    """
    Handles categorical encoding.
    """

    def __init__(self):
        super().__init__()
        self.label_encoders = {}
        logger.info("EncodingHandler Initialized")

    def label_encode(self, data, columns=None):
        """
        Apply Label Encoding.
        """

        data = self.convert_to_dataframe(data)

        if columns is None:
            columns = data.select_dtypes(include=["object"]).columns

        for col in columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            self.label_encoders[col] = le
            logger.info(f"Label Encoding applied on {col}")

        return data

    def one_hot_encode(self, data, columns=None):
        """
        Apply One Hot Encoding.
        """

        data = self.convert_to_dataframe(data)

        if columns is None:
            columns = data.select_dtypes(include=["object"]).columns

        data = pd.get_dummies(data, columns=columns, drop_first=True)

        logger.info("One Hot Encoding applied.")
        return data

    def target_encode(self, data, column, target):
        """
        Simple Target Encoding (Mean Encoding).
        """

        data = self.convert_to_dataframe(data)

        means = data.groupby(column)[target].mean()
        data[column] = data[column].map(means)

        logger.info(f"Target Encoding applied on {column}")
        return data