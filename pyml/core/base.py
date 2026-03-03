"""
PyML Core Base Module
Author: Vaibhav Arun Patil
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from pyml.config import config
from pyml.logger import logger


class BaseModule:
    """
    Base class for all PyML modules.
    """

    def __init__(self):
        self.config = config
        logger.info("BaseModule initialized")

    def validate_data(self, data):

        if data is None:
            raise ValueError("Input data cannot be None.")

        if not isinstance(data, (pd.DataFrame, np.ndarray)):
            raise ValueError("Data must be DataFrame or ndarray.")

        if isinstance(data, pd.DataFrame) and data.empty:
            raise ValueError("DataFrame is empty.")

        return True

    def convert_to_dataframe(self, data):

        if isinstance(data, np.ndarray):
            return pd.DataFrame(data)

        return data

    def get_missing_summary(self, data):

        data = self.convert_to_dataframe(data)
        return data.isnull().sum()

    def check_shape(self, data):

        data = self.convert_to_dataframe(data)
        return data.shape