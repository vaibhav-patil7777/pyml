"""
PyML - A Complete Data Science, Machine Learning & NLP Framework
Author: Vaibhav Arun Patil
License: MIT
"""

__version__ = "0.1.0"

# Import main engine classes (will be created in next files)
# These imports will work once modules are added

try:
    from .pipeline.auto_pipeline import AutoPipeline
except ImportError:
    AutoPipeline = None

try:
    from .models.model_selector import ModelSelector
except ImportError:
    ModelSelector = None

try:
    from .cleaning.missing_values import MissingValueHandler
except ImportError:
    MissingValueHandler = None


class PyML:
    """
    Main User Interface Class
    Provides simple access to all modules.
    """

    def __init__(self, verbose=True):
        self.verbose = verbose

    def info(self):
        """Library Information"""
        return {
            "name": "PyML",
            "version": __version__,
            "description": "Complete Data Scince + ML + NLP Framework",
        }

    def auto_train(self, data, target):
        """
        Automatic End-to-End Training Pipeline
        """
        if AutoPipeline is None:
            raise ImportError("AutoPipeline module not found yet.")

        pipeline = AutoPipeline(verbose=self.verbose)
        return pipeline.run(data, target)

    def list_models(self):
        """List supported models"""
        if ModelSelector is None:
            raise ImportError("ModelSelector module not found yet.")

        selector = ModelSelector()
        return selector.get_supported_models()