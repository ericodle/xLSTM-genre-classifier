"""
GenreDiscern - Music Genre Classification System

A comprehensive music genre classification system using deep learning
with support for multiple neural network architectures.
"""

__version__ = "0.1.0"
__author__ = "Eric and Rebecca"

from .core.config import Config
from .models import get_model
from .training.train import ModelTrainer

__all__ = [
    "Config",
    "get_model",
    "ModelTrainer",
]
