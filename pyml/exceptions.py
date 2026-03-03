"""
PyML Custom Exception Handling
Author: Vaibhav Arun Patil
Version: 0.1.0
"""


class PyMLBaseException(Exception):
    """
    Base Exception for all PyML errors.
    """
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class DataValidationError(PyMLBaseException):
    """
    Raised when dataset is invalid or corrupted.
    """
    pass


class MissingValueError(PyMLBaseException):
    """
    Raised when missing value handling fails.
    """
    pass


class ModelTrainingError(PyMLBaseException):
    """
    Raised when model training fails.
    """
    pass


class FeatureEngineeringError(PyMLBaseException):
    """
    Raised when feature engineering fails.
    """
    pass


class NLPProcessingError(PyMLBaseException):
    """
    Raised when NLP processing fails.
    """
    pass


class ConfigurationError(PyMLBaseException):
    """
    Raised when configuration settings are invalid.
    """
    pass