"""
Utility functions for GenreDiscern.
"""

import logging
import os
import random
from typing import Optional

import torch


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger("GenreDiscern")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Only add handlers if they don't already exist
    if not logger.handlers:
        # Create formatter
        formatter = logging.Formatter(log_format)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Create file handler if log_file is specified
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def get_device(device_preference: str = "auto") -> torch.device:
    """Get the best available device for PyTorch operations."""
    if device_preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        else:
            return torch.device("cpu")
    elif device_preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


def set_random_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_directory(path: str) -> None:
    """Ensure a directory exists, create if it doesn't."""
    os.makedirs(path, exist_ok=True)
