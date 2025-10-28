"""
Core module for GenreDiscern.
"""

from .config import Config
from .utils import get_device, setup_logging

# Optional imports to avoid dependency issues
try:
    from .data_loader import DataLoader
except ImportError:
    DataLoader = None
from .constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_CLASSES,
    MFCC_COEFFICIENTS,
)

__all__ = [
    "Config",
    "DataLoader",
    "setup_logging",
    "get_device",
    "MFCC_COEFFICIENTS",
    "DEFAULT_NUM_CLASSES",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_HIDDEN_SIZE",
    "DEFAULT_LEARNING_RATE",
]
