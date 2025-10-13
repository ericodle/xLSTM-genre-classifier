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
    MIN_AUDIO_DURATION,
    MAX_AUDIO_DURATION,
    DEFAULT_DEVICE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_PIN_MEMORY,
    DEFAULT_CHECKPOINT_INTERVAL,
    DEFAULT_LOG_INTERVAL,
)
from .model_defaults import DEFAULTS, get_defaults


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

    batch_size: int = DEFAULTS.batch_size
    hidden_size: int = DEFAULTS.hidden_size
    num_layers: int = DEFAULTS.num_layers
    dropout: float = DEFAULTS.dropout
    learning_rate: float = DEFAULTS.learning_rate
    max_epochs: int = DEFAULTS.max_epochs
    early_stopping_patience: int = DEFAULTS.early_stopping_patience
    validation_split: float = DEFAULTS.validation_split
    optimizer: str = DEFAULTS.optimizer
    loss_function: str = DEFAULTS.loss_function
    weight_decay: float = DEFAULTS.weight_decay
    lr_scheduler: bool = DEFAULTS.lr_scheduler
    class_weight: str = DEFAULTS.class_weight
    # Optional initializer (none|xavier|kaiming|orthogonal|rnn). None by default
    init: Optional[str] = None
    
    # CNN-specific parameters
    num_classes: int = 10  # Default for GTZAN, will be auto-detected from data
    conv_layers: int = DEFAULTS.conv_layers
    base_filters: int = DEFAULTS.base_filters
    kernel_size: int = DEFAULTS.kernel_size
    pool_size: int = DEFAULTS.pool_size
    fc_hidden: int = DEFAULTS.fc_hidden
    
    # Transformer-specific parameters
    num_heads: int = DEFAULTS.num_heads
    ff_dim: int = DEFAULTS.ff_dim


@dataclass
class TrainingConfig:
    """Training process configuration."""

    device: str = DEFAULT_DEVICE
    num_workers: int = DEFAULT_NUM_WORKERS
    pin_memory: bool = DEFAULT_PIN_MEMORY
    save_best_model: bool = DEFAULTS.save_best_model
    save_checkpoints: bool = DEFAULTS.save_checkpoints
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL
    log_interval: int = DEFAULT_LOG_INTERVAL
    random_seed: int = DEFAULTS.random_seed
    early_stopping: bool = DEFAULTS.early_stopping
    patience: int = DEFAULTS.early_stopping_patience
    improvement_threshold: float = DEFAULTS.improvement_threshold
    improvement_window: int = DEFAULTS.early_stopping_patience
    gradient_clip_norm: Optional[float] = None


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

    def optimize_for_dataset(self, dataset_type: str, model_type: str = "GRU") -> None:
        """Optimize configuration for specific dataset type."""
        optimized_defaults = get_defaults(model_type, dataset_type)
        
        # Update model config with optimized defaults
        for key, value in optimized_defaults.items():
            if hasattr(self.model, key):
                setattr(self.model, key, value)
            elif hasattr(self.training, key):
                setattr(self.training, key, value)
        
        # Log the optimized parameters for debugging
        print(f"Optimized parameters for {dataset_type} dataset:")
        for key, value in optimized_defaults.items():
            print(f"  {key}: {value}")

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
