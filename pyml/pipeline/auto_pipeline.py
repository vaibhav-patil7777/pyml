"""
PyML Auto Pipeline Module
Author: Vaibhav Arun Patil
Version: 0.1.0
"""

import numpy as np
import pandas as pd

from ..core.base import BaseModule
from ..cleaning.missing_values import MissingValueHandler
from ..preprocessing.encoding import EncodingHandler
from ..preprocessing.scaling import ScalingHandler
from ..model.model_engine import ModelEngine
from ..evaluation.metrics import Metrics
from ..logger import logger


class AutoPipeline(BaseModule):

    def __init__(self, verbose=True):
        super().__init__()
        self.verbose = verbose

        self.missing_handler = MissingValueHandler()
        self.encoder = EncodingHandler()
        self.scaler = ScalingHandler()
        self.model_engine = ModelEngine()
        self.metrics = Metrics()

        logger.info("AutoPipeline Initialized")

    def run(self, data, target):

        # Convert to DataFrame
        data = self.convert_to_dataframe(data)

        # Separate features and target
        X = data.drop(columns=[target])
        y = data[target]

        # 1. Handle Missing Values
        X = self.missing_handler.handle_missing(X)

        # 2. Encoding
        X = self.encoder.label_encode(X)

        # 3. Scaling
        X = self.scaler.scale(X, method="standard")

        # 4. Detect Problem Type Automatically
        if len(np.unique(y)) <= 20:
            problem_type = "classification"
        else:
            problem_type = "regression"

        # 5. Train Model
        model = self.model_engine.train(X, y, model_name="auto")

        # 6. Predictions
        if problem_type == "classification":
            y_pred = model.predict(X)
            report = self.metrics.classification_report(y, y_pred)

        else:
            y_pred = model.predict(X)
            report = self.metrics.regression_report(y, y_pred)

        logger.info("Auto Pipeline Completed Successfully")

        return {
            "model": model,
            "metrics": report
        }