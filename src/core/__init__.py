"""
Core module for GenreDiscern.
"""

from .config import Config
from .utils import setup_logging, get_device

# Optional imports to avoid dependency issues
try:
    from .data_loader import DataLoader
except ImportError:
    DataLoader = None
from .constants import (
    MFCC_COEFFICIENTS,
    DEFAULT_NUM_CLASSES,
    GTZAN_GENRES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_LEARNING_RATE,
)

__all__ = [
    "Config",
    "DataLoader",
    "setup_logging",
    "get_device",
    "MFCC_COEFFICIENTS",
    "DEFAULT_NUM_CLASSES",
    "GTZAN_GENRES",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_HIDDEN_SIZE",
    "DEFAULT_LEARNING_RATE",
]
