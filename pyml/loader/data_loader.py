"""
PyML Data Loader Module
Author: Vaibhav Arun Patil
Version: 0.1.0
"""

import pandas as pd
import os
from ..core.base import BaseModule
from ..logger import logger
from ..exceptions import DataValidationError


class DataLoader(BaseModule):
    """
    Handles loading of different data formats.
    Supports CSV, Excel, and JSON.
    """

    def __init__(self):
        super().__init__()
        logger.info("DataLoader Initialized")

    def load_csv(self, file_path):
        """
        Load CSV file.
        """
        if not os.path.exists(file_path):
            raise DataValidationError("CSV file not found.")

        data = pd.read_csv(file_path)
        self.validate_data(data)

        logger.info("CSV file loaded successfully.")
        return data

    def load_excel(self, file_path):
        """
        Load Excel file.
        """
        if not os.path.exists(file_path):
            raise DataValidationError("Excel file not found.")

        data = pd.read_excel(file_path)
        self.validate_data(data)

        logger.info("Excel file loaded successfully.")
        return data

    def load_json(self, file_path):
        """
        Load JSON file.
        """
        if not os.path.exists(file_path):
            raise DataValidationError("JSON file not found.")

        data = pd.read_json(file_path)
        self.validate_data(data)

        logger.info("JSON file loaded successfully.")
        return data

    def auto_load(self, file_path):
        """
        Automatically detect file type and load data.
        """
        if not os.path.exists(file_path):
            raise DataValidationError("File not found.")

        extension = os.path.splitext(file_path)[1].lower()

        if extension == ".csv":
            return self.load_csv(file_path)
        elif extension in [".xls", ".xlsx"]:
            return self.load_excel(file_path)
        elif extension == ".json":
            return self.load_json(file_path)
        else:
            raise DataValidationError("Unsupported file format.")