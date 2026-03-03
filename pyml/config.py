"""
PyML Configuration Module
Author: Vaibhav Arun Patil
Version: 1.0.0
"""

import os


class Config:
    """
    Global Configuration for PyML Library
    """

    def __init__(self):
        # General
        self.VERSION = "1.0.0"
        self.AUTHOR = "Vaibhav Arun Patil"

        # Data
        self.TEST_SIZE = 0.2
        self.RANDOM_STATE = 42

        # Missing Values
        self.NUMERICAL_MISSING_STRATEGY = "mean"
        self.CATEGORICAL_MISSING_STRATEGY = "most_frequent"

        # Scaling
        self.DEFAULT_SCALER = "standard"

        # Feature Engineering
        self.APPLY_PCA = False
        self.PCA_COMPONENTS = 0.95

        # Model
        self.AUTO_MODEL_SELECTION = True

        # Logging
        self.ENABLE_LOGGING = True
        self.LOG_FILE = "pyml.log"

    def get_project_root(self):
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Global config instance
config = Config()