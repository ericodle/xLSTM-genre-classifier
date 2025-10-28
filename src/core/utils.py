"""
Utility functions for GenreDiscern.
"""

import logging
import os
import random
from typing import Any, Optional, Tuple

import numpy as np
import torch

from .constants import (
    AUDIO_EXTENSIONS,
    BYTES_PER_GB,
    BYTES_PER_KB,
    BYTES_PER_MB,
    EXIT_FAILURE,
    EXIT_INTERRUPT,
    EXIT_SUCCESS,
    MFCC_COEFFICIENTS,
)


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
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_directory(path: str) -> None:
    """Ensure a directory exists, create if it doesn't."""
    os.makedirs(path, exist_ok=True)


def get_file_extension(file_path: str) -> str:
    """Get file extension from file path."""
    return os.path.splitext(file_path)[1].lower()


def is_audio_file(file_path: str) -> bool:
    """Check if file is an audio file based on extension."""
    return get_file_extension(file_path) in AUDIO_EXTENSIONS


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to [-1, 1] range."""
    if np.max(np.abs(audio)) > 0:
        return np.asarray(audio / np.max(np.abs(audio)))
    return audio


def pad_or_truncate(audio: np.ndarray, target_length: int) -> np.ndarray:
    """Pad or truncate audio to target length."""
    if len(audio) > target_length:
        return audio[:target_length]
    elif len(audio) < target_length:
        padding = target_length - len(audio)
        return np.pad(audio, (0, padding), mode="constant")
    return audio


def calculate_mfcc_shape(
    audio_length: int, sample_rate: int, n_fft: int, hop_length: int
) -> Tuple[int, int]:
    """Calculate the expected shape of MFCC features."""
    num_frames = 1 + (audio_length - n_fft) // hop_length
    return (num_frames, MFCC_COEFFICIENTS)


def validate_audio_parameters(sample_rate: int, n_fft: int, hop_length: int) -> bool:
    """Validate audio processing parameters."""
    if sample_rate <= 0:
        return False
    if n_fft <= 0 or n_fft % 2 != 0:
        return False
    if hop_length <= 0 or hop_length > n_fft:
        return False
    return True


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"


def format_file_size(bytes_size: int) -> str:
    """Format file size in bytes to human readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < BYTES_PER_KB:
            return f"{bytes_size:.1f} {unit}"
        bytes_size = int(bytes_size / BYTES_PER_KB)
    return f"{bytes_size:.1f} TB"
