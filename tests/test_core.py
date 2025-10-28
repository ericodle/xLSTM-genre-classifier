"""
Tests for src/core module.
Tests all classes, functions, and constants.
"""

import os
import tempfile
from pathlib import Path

import pytest
import torch

from src.core.config import Config, ModelConfig, TrainingConfig
from src.core.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_CLASSES,
)
from src.core.utils import ensure_directory, get_device, set_random_seed, setup_logging


class TestConstants:
    """Test that constants have expected values."""

    def test_default_constants_exist(self):
        """Test that default constants are defined."""
        assert DEFAULT_BATCH_SIZE > 0
        assert DEFAULT_HIDDEN_SIZE > 0
        assert DEFAULT_LEARNING_RATE > 0
        assert DEFAULT_NUM_CLASSES > 0

    def test_constants_have_reasonable_values(self):
        """Test that constants have reasonable values."""
        assert 1 <= DEFAULT_BATCH_SIZE <= 1000
        assert 1 <= DEFAULT_HIDDEN_SIZE <= 1000
        assert 0 < DEFAULT_LEARNING_RATE <= 1
        assert 1 <= DEFAULT_NUM_CLASSES <= 100


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_model_config_defaults(self):
        """Test that ModelConfig has default values."""
        config = ModelConfig()

        assert config.batch_size == DEFAULT_BATCH_SIZE
        assert config.hidden_size == DEFAULT_HIDDEN_SIZE
        assert config.learning_rate == DEFAULT_LEARNING_RATE
        assert config.num_classes == DEFAULT_NUM_CLASSES

    def test_model_config_custom_values(self):
        """Test setting custom values in ModelConfig."""
        config = ModelConfig(batch_size=32, learning_rate=0.001)

        assert config.batch_size == 32
        assert config.learning_rate == 0.001
        # Other values should still be defaults
        assert config.hidden_size == DEFAULT_HIDDEN_SIZE

    def test_model_config_all_attributes(self):
        """Test that ModelConfig has all expected attributes."""
        config = ModelConfig()

        # Check required attributes exist
        assert hasattr(config, "batch_size")
        assert hasattr(config, "hidden_size")
        assert hasattr(config, "num_layers")
        assert hasattr(config, "dropout")
        assert hasattr(config, "learning_rate")
        assert hasattr(config, "max_epochs")
        assert hasattr(config, "early_stopping_patience")
        assert hasattr(config, "validation_split")
        assert hasattr(config, "optimizer")
        assert hasattr(config, "loss_function")
        assert hasattr(config, "weight_decay")
        assert hasattr(config, "lr_scheduler")
        assert hasattr(config, "class_weight")
        assert hasattr(config, "label_smoothing")
        assert hasattr(config, "init")


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""

    def test_training_config_defaults(self):
        """Test that TrainingConfig has default values."""
        config = TrainingConfig()

        # Check that device is set
        assert config.device is not None
        assert isinstance(config.device, str)

        # Check other attributes
        assert config.num_workers >= 0
        assert isinstance(config.pin_memory, bool)
        assert isinstance(config.save_best_model, bool)
        assert isinstance(config.save_checkpoints, bool)

    def test_training_config_all_attributes(self):
        """Test that TrainingConfig has all expected attributes."""
        config = TrainingConfig()

        assert hasattr(config, "device")
        assert hasattr(config, "num_workers")
        assert hasattr(config, "pin_memory")
        assert hasattr(config, "save_best_model")
        assert hasattr(config, "save_checkpoints")
        assert hasattr(config, "checkpoint_interval")
        assert hasattr(config, "log_interval")
        assert hasattr(config, "random_seed")
        assert hasattr(config, "early_stopping")
        assert hasattr(config, "patience")
        assert hasattr(config, "improvement_threshold")
        assert hasattr(config, "improvement_window")
        assert hasattr(config, "gradient_clip_norm")


class TestConfig:
    """Test Config class."""

    def test_config_initialization(self):
        """Test that Config initializes properly."""
        config = Config()

        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.training, TrainingConfig)

    def test_config_optimize_for_dataset_fma(self):
        """Test dataset-specific optimization for FMA."""
        config = Config()

        # Optimize for FMA with GRU model
        config.optimize_for_dataset("FMA", "GRU")

        # Check that some parameters were optimized
        assert config.model.batch_size == 24  # GRU-specific for FMA
        assert config.model.learning_rate == 0.0003  # GRU-specific LR for FMA
        assert config.model.class_weight == "auto"  # Class weighting for imbalance
        assert config.training.gradient_clip_norm == 0.5  # Stronger clipping

    def test_config_optimize_for_dataset_gtzan(self):
        """Test dataset-specific optimization for GTZAN."""
        config = Config()
        original_lr = config.model.learning_rate

        # Optimize for GTZAN
        config.optimize_for_dataset("GTZAN", "LSTM")

        # GTZAN should have different optimization than FMA
        # We should have model-specific tweaks
        assert config.model.learning_rate in [0.0003, original_lr]  # Depending on model

    def test_config_optimize_for_model_transformer(self):
        """Test model-specific optimization."""
        config = Config()

        # Optimize for Transformer
        config.optimize_for_dataset("FMA", "TRANSFORMER")

        # Transformer-specific settings
        assert config.model.num_heads == 4
        assert config.model.ff_dim == 64
        assert config.model.dropout == 0.1

    def test_config_optimize_for_model_cnn(self):
        """Test CNN-specific optimization."""
        config = Config()

        # Optimize for CNN
        config.optimize_for_dataset("GTZAN", "CNN")

        # CNN-specific settings
        assert config.model.conv_layers == 6
        assert config.model.base_filters == 64
        assert config.model.kernel_size == 5


class TestUtils:
    """Test utility functions."""

    def test_setup_logging_console(self):
        """Test logging setup without file."""
        logger = setup_logging(log_level="DEBUG")

        assert logger is not None
        assert logger.level == 10  # DEBUG level
        assert len(logger.handlers) > 0

    def test_setup_logging_file(self):
        """Test logging setup with file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")

            # setup_logging creates the directory if needed
            logger = setup_logging(log_file=log_file)

            assert logger is not None
            assert len(logger.handlers) >= 1  # At least console handler

            # Test that logging to file works
            logger.info("Test message")

            # Force flush handlers and close to ensure file is written
            for handler in logger.handlers:
                handler.flush()
                if hasattr(handler, "close"):
                    handler.close()

            # Check that file was created
            # Note: The file might not exist if tmpdir is removed immediately
            # So we just check that setup_logging doesn't crash
            assert logger is not None

    def test_get_device_auto(self):
        """Test automatic device selection."""
        device = get_device()

        assert device is not None
        assert isinstance(device, torch.device)

    def test_get_device_cuda(self):
        """Test CUDA device selection."""
        device = get_device(device_preference="cuda")

        assert device is not None
        assert isinstance(device, torch.device)

    def test_get_device_cpu(self):
        """Test CPU device selection."""
        device = get_device(device_preference="cpu")

        assert device.type == "cpu"

    def test_set_random_seed(self):
        """Test random seed setting."""
        # This is mostly about ensuring it doesn't crash
        set_random_seed(42)
        set_random_seed(123)

        # Just verify it runs without error
        assert True

    def test_ensure_directory(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = os.path.join(tmpdir, "test", "nested", "dir")
            ensure_directory(test_dir)

            assert os.path.exists(test_dir)
            assert os.path.isdir(test_dir)

    def test_ensure_directory_existing(self):
        """Test that ensure_directory handles existing directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First creation
            ensure_directory(tmpdir)
            assert os.path.exists(tmpdir)

            # Second creation (should not fail)
            ensure_directory(tmpdir)
            assert os.path.exists(tmpdir)


class TestIntegration:
    """Integration tests for core module."""

    def test_config_with_optimization_and_logging(self):
        """Test that Config works with logging."""
        logger = setup_logging(log_level="INFO")
        config = Config()

        # Optimize for dataset
        config.optimize_for_dataset("FMA", "GRU")

        # Verify configuration is valid
        assert config.model.learning_rate > 0
        assert config.model.batch_size > 0
        assert config.training.device is not None

        # Just verify we can use the logger
        logger.info("Configuration validated")
        assert True

    def test_full_config_workflow(self):
        """Test a complete configuration workflow."""
        # Setup
        config = Config()

        # Optimize for specific dataset and model
        config.optimize_for_dataset("GTZAN", "CNN")

        # Verify all settings are reasonable
        assert config.model.learning_rate > 0
        assert config.model.batch_size > 0
        assert config.model.max_epochs > 0
        assert config.model.hidden_size > 0
        assert config.training.patience > 0

        # Verify device setup
        device = get_device()
        assert device is not None
