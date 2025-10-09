"""
Configuration management for GenreDiscern.
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
import json

from .constants import (
    SAMPLE_RATE,
    HOP_LENGTH,
    N_FFT,
    DEFAULT_N_MFCC,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_VALIDATION_SPLIT,
    DEFAULT_TEST_SIZE,
    DEFAULT_RANDOM_SEED,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    MIN_AUDIO_DURATION,
    MAX_AUDIO_DURATION,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_NUM_LAYERS,
    DEFAULT_DROPOUT,
    DEFAULT_OPTIMIZER,
    DEFAULT_LOSS_FUNCTION,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_LR_SCHEDULER,
    DEFAULT_DEVICE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_PIN_MEMORY,
    DEFAULT_SAVE_BEST_MODEL,
    DEFAULT_SAVE_CHECKPOINTS,
    DEFAULT_CHECKPOINT_INTERVAL,
    DEFAULT_LOG_INTERVAL,
    DEFAULT_EARLY_STOPPING,
    DEFAULT_PATIENCE,
)


@dataclass
class AudioConfig:
    """Audio processing configuration."""

    sample_rate: int = SAMPLE_RATE
    hop_length: int = HOP_LENGTH
    n_fft: int = N_FFT
    n_mfcc: int = DEFAULT_N_MFCC  # Now configurable!
    min_duration: float = MIN_AUDIO_DURATION
    max_duration: float = MAX_AUDIO_DURATION


@dataclass
class ModelConfig:
    """Model training configuration."""

    batch_size: int = DEFAULT_BATCH_SIZE
    hidden_size: int = DEFAULT_HIDDEN_SIZE
    num_layers: int = DEFAULT_NUM_LAYERS
    dropout: float = DEFAULT_DROPOUT
    learning_rate: float = DEFAULT_LEARNING_RATE
    max_epochs: int = DEFAULT_MAX_EPOCHS
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE
    validation_split: float = DEFAULT_VALIDATION_SPLIT
    optimizer: str = DEFAULT_OPTIMIZER
    loss_function: str = DEFAULT_LOSS_FUNCTION
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    lr_scheduler: bool = DEFAULT_LR_SCHEDULER
    class_weight: str = "none"  # "none", "auto", or comma-separated weights
    
    # CNN-specific parameters
    num_classes: int = 10  # Default for GTZAN, will be auto-detected from data
    conv_layers: int = 3
    base_filters: int = 16
    kernel_size: int = 3
    pool_size: int = 2
    fc_hidden: int = 64


@dataclass
class TrainingConfig:
    """Training process configuration."""

    device: str = DEFAULT_DEVICE
    num_workers: int = DEFAULT_NUM_WORKERS
    pin_memory: bool = DEFAULT_PIN_MEMORY
    save_best_model: bool = DEFAULT_SAVE_BEST_MODEL
    save_checkpoints: bool = DEFAULT_SAVE_CHECKPOINTS
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL
    log_interval: int = DEFAULT_LOG_INTERVAL
    random_seed: int = DEFAULT_RANDOM_SEED
    early_stopping: bool = DEFAULT_EARLY_STOPPING
    patience: int = DEFAULT_PATIENCE
    improvement_threshold: float = 0.001
    improvement_window: int = DEFAULT_PATIENCE


@dataclass
class PathConfig:
    """Path configuration."""

    data_dir: str = "data"
    models_dir: str = "models"
    output_dir: str = "output"
    logs_dir: str = "logs"
    cache_dir: str = "cache"


class Config:
    """Main configuration class for GenreDiscern."""

    def __init__(self, config_path: Optional[str] = None):
        self.audio = AudioConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.paths = PathConfig()

        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)

    def load_from_file(self, config_path: str) -> None:
        """Load configuration from JSON file."""
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)

            # Update nested configurations
            if "audio" in config_data:
                for key, value in config_data["audio"].items():
                    if hasattr(self.audio, key):
                        setattr(self.audio, key, value)

            if "model" in config_data:
                for key, value in config_data["model"].items():
                    if hasattr(self.model, key):
                        setattr(self.model, key, value)

            if "training" in config_data:
                for key, value in config_data["training"].items():
                    if hasattr(self.training, key):
                        setattr(self.training, key, value)

            if "paths" in config_data:
                for key, value in config_data["paths"].items():
                    if hasattr(self.paths, key):
                        setattr(self.paths, key, value)

        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")

    def save_to_file(self, config_path: str) -> None:
        """Save configuration to JSON file."""
        try:
            config_data = {
                "audio": self.audio.__dict__,
                "model": self.model.__dict__,
                "training": self.training.__dict__,
                "paths": self.paths.__dict__,
            }

            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2)

        except Exception as e:
            print(f"Warning: Could not save config to {config_path}: {e}")

    def get_audio_config(self) -> Dict[str, Any]:
        """Get audio configuration as dictionary."""
        return self.audio.__dict__

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration as dictionary."""
        return self.model.__dict__

    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration as dictionary."""
        return self.training.__dict__

    def get_paths_config(self) -> Dict[str, Any]:
        """Get paths configuration as dictionary."""
        return self.paths.__dict__


# Default configuration instance
default_config = Config()
