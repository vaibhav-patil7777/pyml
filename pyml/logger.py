"""
PyML Logging Module
Author: Vaibhav Arun Patil
Version: 1.0.0
"""

import logging


def get_logger():
    from pyml.config import config   # Lazy import (IMPORTANT)

    logger = logging.getLogger("PyML")

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    # Console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File
    if config.ENABLE_LOGGING:
        file_handler = logging.FileHandler(config.LOG_FILE)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = get_logger()