"""
Unit tests for core configuration module.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path

# Add src directory to path for imports
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.config import Config, AudioConfig, ModelConfig, TrainingConfig, PathConfig
from core.constants import (
    MFCC_COEFFICIENTS,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SONG_LENGTH,
    DEFAULT_N_FFT,
    DEFAULT_HOP_LENGTH,
    DEFAULT_SEGMENT_LENGTH,
    DEFAULT_BATCH_SIZE,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_NUM_LAYERS,
    DEFAULT_DROPOUT,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_VALIDATION_SPLIT,
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
    DEFAULT_RANDOM_SEED,
    DEFAULT_EARLY_STOPPING,
    DEFAULT_PATIENCE,
    DEFAULT_N_MFCC,
    MIN_AUDIO_DURATION,
    MAX_AUDIO_DURATION,
)


class TestAudioConfig:
    """Test AudioConfig class."""

    def test_default_values(self):
        """Test default values are set correctly."""
        config = AudioConfig()

        assert config.sample_rate == DEFAULT_SAMPLE_RATE
        assert config.n_mfcc == DEFAULT_N_MFCC
        assert config.n_fft == DEFAULT_N_FFT
        assert config.hop_length == DEFAULT_HOP_LENGTH
        assert config.min_duration == MIN_AUDIO_DURATION
        assert config.max_duration == MAX_AUDIO_DURATION

    def test_custom_values(self):
        """Test custom values can be set."""
        config = AudioConfig(sample_rate=44100, n_mfcc=20, min_duration=10.0)

        assert config.sample_rate == 44100
        assert config.n_mfcc == 20
        assert config.min_duration == 10.0
        assert config.n_fft == DEFAULT_N_FFT  # Should use default
        assert config.hop_length == DEFAULT_HOP_LENGTH  # Should use default
        assert config.min_duration == 10.0  # Custom value


class TestModelConfig:
    """Test ModelConfig class."""

    def test_default_values(self):
        """Test default values are set correctly."""
        config = ModelConfig()

        assert config.batch_size == DEFAULT_BATCH_SIZE
        assert config.hidden_size == DEFAULT_HIDDEN_SIZE
        assert config.num_layers == DEFAULT_NUM_LAYERS
        assert config.dropout == DEFAULT_DROPOUT
        assert config.learning_rate == DEFAULT_LEARNING_RATE
        assert config.max_epochs == DEFAULT_MAX_EPOCHS
        assert config.early_stopping_patience == DEFAULT_EARLY_STOPPING_PATIENCE
        assert config.validation_split == DEFAULT_VALIDATION_SPLIT
        assert config.optimizer == DEFAULT_OPTIMIZER
        assert config.loss_function == DEFAULT_LOSS_FUNCTION
        assert config.weight_decay == DEFAULT_WEIGHT_DECAY
        assert config.lr_scheduler == DEFAULT_LR_SCHEDULER

    def test_custom_values(self):
        """Test custom values can be set."""
        config = ModelConfig(batch_size=128, hidden_size=64, learning_rate=0.001)

        assert config.batch_size == 128
        assert config.hidden_size == 64
        assert config.learning_rate == 0.001
        assert config.num_layers == DEFAULT_NUM_LAYERS  # Should use default
        assert config.dropout == DEFAULT_DROPOUT  # Should use default


class TestTrainingConfig:
    """Test TrainingConfig class."""

    def test_default_values(self):
        """Test default values are set correctly."""
        config = TrainingConfig()

        assert config.device == DEFAULT_DEVICE
        assert config.num_workers == DEFAULT_NUM_WORKERS
        assert config.pin_memory == DEFAULT_PIN_MEMORY
        assert config.save_best_model == DEFAULT_SAVE_BEST_MODEL
        assert config.save_checkpoints == DEFAULT_SAVE_CHECKPOINTS
        assert config.checkpoint_interval == DEFAULT_CHECKPOINT_INTERVAL
        assert config.log_interval == DEFAULT_LOG_INTERVAL
        assert config.random_seed == DEFAULT_RANDOM_SEED
        assert config.early_stopping == DEFAULT_EARLY_STOPPING
        assert config.patience == DEFAULT_PATIENCE

    def test_custom_values(self):
        """Test custom values can be set."""
        config = TrainingConfig(device="cuda", num_workers=8, random_seed=123)

        assert config.device == "cuda"
        assert config.num_workers == 8
        assert config.random_seed == 123
        assert config.pin_memory == DEFAULT_PIN_MEMORY  # Should use default


class TestPathConfig:
    """Test PathConfig class."""

    def test_default_values(self):
        """Test default values are set correctly."""
        config = PathConfig()

        assert config.data_dir == "data"
        assert config.models_dir == "models"
        assert config.output_dir == "output"
        assert config.logs_dir == "logs"
        assert config.cache_dir == "cache"

    def test_custom_values(self):
        """Test custom values can be set."""
        config = PathConfig(data_dir="/custom/data", output_dir="/custom/output")

        assert config.data_dir == "/custom/data"
        assert config.output_dir == "/custom/output"
        assert config.models_dir == "models"  # Should use default


class TestConfig:
    """Test main Config class."""

    def test_default_initialization(self):
        """Test default initialization."""
        config = Config()

        # Check that all sub-configs are created
        assert isinstance(config.audio, AudioConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.paths, PathConfig)

        # Check default values
        assert config.audio.n_mfcc == DEFAULT_N_MFCC
        assert config.model.batch_size == DEFAULT_BATCH_SIZE
        assert config.training.random_seed == DEFAULT_RANDOM_SEED

    def test_load_from_file(self):
        """Test loading configuration from file."""
        # Create temporary config file
        config_data = {
            "audio": {"sample_rate": 48000, "n_mfcc": 26},
            "model": {"batch_size": 256, "learning_rate": 0.0001},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            config = Config(config_path)

            # Check that values were loaded
            assert config.audio.sample_rate == 48000
            assert config.audio.n_mfcc == 26
            assert config.model.batch_size == 256
            assert config.model.learning_rate == 0.0001

            # Check that other values remain default
            assert config.audio.n_fft == DEFAULT_N_FFT
            assert config.model.hidden_size == DEFAULT_HIDDEN_SIZE

        finally:
            # Clean up
            os.unlink(config_path)

    def test_save_to_file(self):
        """Test saving configuration to file."""
        config = Config()

        # Modify some values
        config.audio.sample_rate = 48000
        config.model.batch_size = 256

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_path = f.name

        try:
            config.save_to_file(config_path)

            # Load back and verify
            loaded_config = Config(config_path)
            assert loaded_config.audio.sample_rate == 48000
            assert loaded_config.model.batch_size == 256

        finally:
            # Clean up
            os.unlink(config_path)

    def test_get_config_methods(self):
        """Test getter methods for configurations."""
        config = Config()

        # Test audio config getter
        audio_config = config.get_audio_config()
        assert isinstance(audio_config, dict)
        assert audio_config["n_mfcc"] == DEFAULT_N_MFCC

        # Test model config getter
        model_config = config.get_model_config()
        assert isinstance(model_config, dict)
        assert model_config["batch_size"] == DEFAULT_BATCH_SIZE

        # Test training config getter
        training_config = config.get_training_config()
        assert isinstance(training_config, dict)
        assert training_config["random_seed"] == DEFAULT_RANDOM_SEED

        # Test paths config getter
        paths_config = config.get_paths_config()
        assert isinstance(paths_config, dict)
        assert paths_config["data_dir"] == "data"

    def test_invalid_config_file(self):
        """Test handling of invalid config file."""
        # Should not raise exception, just log warning
        config = Config("nonexistent_file.json")

        # Should use default values
        assert config.audio.n_mfcc == DEFAULT_N_MFCC
        assert config.model.batch_size == DEFAULT_BATCH_SIZE


if __name__ == "__main__":
    pytest.main([__file__])
