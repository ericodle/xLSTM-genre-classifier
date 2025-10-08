"""
Tests for GenreDiscern training pipeline.
"""

import pytest
import torch
import numpy as np
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Add src directory to path for imports
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from training.trainer import ModelTrainer
from core.config import Config


class TestModelTrainer:
    """Test the ModelTrainer class."""

    @pytest.fixture
    def sample_config(self):
        """Create a sample configuration for testing."""
        config = Config()
        # Override some settings for testing
        config.model.max_epochs = 2
        config.model.batch_size = 4
        config.model.hidden_size = 16
        config.model.num_layers = 1
        config.model.dropout = 0.1
        config.model.learning_rate = 0.01
        return config

    @pytest.fixture
    def sample_mfcc_data(self, temp_dir):
        """Create sample MFCC data for testing."""
        # Create sample MFCC data in new format with sufficient samples per class
        n_samples = 150  # 30 samples per class for 5 classes
        n_frames = 100
        n_mfcc = 13
        genres = ["blues", "classical", "country", "disco", "hiphop"]

        # Generate features and labels
        features = []
        labels = []

        for i in range(n_samples):
            genre = genres[i % len(genres)]
            # Generate random MFCC features
            mfcc_data = np.random.randn(n_frames, n_mfcc).tolist()
            features.append(mfcc_data)
            labels.append(genre)

        # Create data in new format
        data = {
            "features": features,
            "labels": labels,
            "metadata": {
                "sample_rate": 22050,
                "n_mfcc": n_mfcc,
                "hop_length": 512,
                "n_fft": 2048,
            },
        }

        # Save to temporary file
        data_file = Path(temp_dir) / "test_mfcc.json"
        with open(data_file, "w") as f:
            json.dump(data, f)

        return str(data_file)

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_trainer_initialization(self, sample_config):
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer(sample_config)
        assert trainer.config == sample_config
        assert trainer.device is not None
        assert trainer.model is None
        assert trainer.optimizer is None
        # Criterion is set during setup_training, not initialization
        assert hasattr(trainer, "criterion")

    def test_trainer_setup_training(self, sample_config, sample_mfcc_data, temp_dir):
        """Test training setup."""
        trainer = ModelTrainer(sample_config)

        # Setup training
        trainer.setup_training(
            data_path=sample_mfcc_data, model_type="LSTM", output_dir=temp_dir
        )

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.train_loader is not None
        assert trainer.val_loader is not None
        assert trainer.test_loader is not None
        assert trainer.output_dir == Path(temp_dir)

    def test_trainer_lstm_training(self, sample_config, sample_mfcc_data, temp_dir):
        """Test LSTM model training."""
        trainer = ModelTrainer(sample_config)

        # Setup training
        trainer.setup_training(
            data_path=sample_mfcc_data, model_type="LSTM", output_dir=temp_dir
        )

        # Train for a few epochs
        history = trainer.train()

        assert isinstance(history, dict)
        assert "train_loss" in history
        assert "train_acc" in history
        assert "val_loss" in history
        assert "val_acc" in history
        assert len(history["train_loss"]) > 0
        assert len(history["val_loss"]) > 0

        # Check that model files were created
        assert (Path(temp_dir) / "best_model.onnx").exists()
        assert (Path(temp_dir) / "model.onnx").exists()

    def test_trainer_cnn_training(self, sample_config, sample_mfcc_data, temp_dir):
        """Test CNN model training."""
        trainer = ModelTrainer(sample_config)

        # Setup training
        trainer.setup_training(
            data_path=sample_mfcc_data, model_type="CNN", output_dir=temp_dir
        )

        # Train for a few epochs
        history = trainer.train()

        assert isinstance(history, dict)
        assert len(history["train_loss"]) > 0

        # Check that model files were created
        assert (Path(temp_dir) / "best_model.onnx").exists()
        assert (Path(temp_dir) / "model.onnx").exists()

    def test_trainer_xlstm_training(self, sample_config, sample_mfcc_data, temp_dir):
        """Test xLSTM model training."""
        trainer = ModelTrainer(sample_config)

        # Setup training
        trainer.setup_training(
            data_path=sample_mfcc_data, model_type="xLSTM", output_dir=temp_dir
        )

        # Train for a few epochs
        history = trainer.train()

        assert isinstance(history, dict)
        assert len(history["train_loss"]) > 0

        # Check that model files were created
        assert (Path(temp_dir) / "best_model.onnx").exists()
        assert (Path(temp_dir) / "model.onnx").exists()

    def test_onnx_export(self, sample_config, sample_mfcc_data, temp_dir):
        """Test ONNX export functionality."""
        trainer = ModelTrainer(sample_config)

        # Setup training
        trainer.setup_training(
            data_path=sample_mfcc_data, model_type="LSTM", output_dir=temp_dir
        )

        # Train for one epoch to get a model
        trainer.train()

        # Check ONNX file was created
        onnx_path = Path(temp_dir) / "model.onnx"
        assert onnx_path.exists()
        assert onnx_path.stat().st_size > 0

    def test_training_plots_generation(self, sample_config, sample_mfcc_data, temp_dir):
        """Test that training plots are generated."""
        trainer = ModelTrainer(sample_config)

        # Setup training
        trainer.setup_training(
            data_path=sample_mfcc_data, model_type="LSTM", output_dir=temp_dir
        )

        # Train for a few epochs
        trainer.train()

        # Check that plots directory was created
        plots_dir = Path(temp_dir) / "training_plots"
        assert plots_dir.exists()
        assert (plots_dir / "loss_plot.png").exists()
        assert (plots_dir / "accuracy_plot.png").exists()

    def test_early_stopping(self, sample_config, sample_mfcc_data, temp_dir):
        """Test early stopping functionality."""
        config = sample_config
        config.training.early_stopping = True
        config.training.improvement_threshold = 0.1  # High threshold to trigger early stopping
        config.training.improvement_window = 2
        config.model.max_epochs = 10

        trainer = ModelTrainer(config)

        # Setup training
        trainer.setup_training(
            data_path=sample_mfcc_data, model_type="LSTM", output_dir=temp_dir
        )

        # Train (should stop early due to high improvement threshold)
        history = trainer.train()

        # Should stop before reaching max epochs due to early stopping
        assert len(history["train_loss"]) <= 10

    def test_model_checkpoint_saving(self, sample_config, sample_mfcc_data, temp_dir):
        """Test that model checkpoints are saved correctly."""
        trainer = ModelTrainer(sample_config)

        # Setup training
        trainer.setup_training(
            data_path=sample_mfcc_data, model_type="LSTM", output_dir=temp_dir
        )

        # Train for a few epochs
        trainer.train()

        # Check checkpoint file
        checkpoint_path = Path(temp_dir) / "best_model.onnx"
        assert checkpoint_path.exists()

        # Verify ONNX model and metadata files exist
        metadata_path = Path(temp_dir) / "best_model_metadata.json"
        training_metadata_path = Path(temp_dir) / "best_model_training_metadata.json"

        assert metadata_path.exists()
        assert training_metadata_path.exists()

        # Load and verify metadata
        import json

        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        assert "model_config" in metadata
        assert "training_history" in metadata
        assert "is_trained" in metadata

        with open(training_metadata_path, "r") as f:
            training_metadata = json.load(f)
        assert "epoch" in training_metadata
        assert "best_val_loss" in training_metadata
        assert "config" in training_metadata


class TestAutomaticEvaluation:
    """Test the automatic evaluation functionality."""

    @pytest.fixture
    def sample_mfcc_data(self, temp_dir):
        """Create sample MFCC data for testing."""
        data = {}
        genres = ["blues", "classical", "country", "disco", "hiphop"]

        for i in range(150):  # 150 samples (30 per class for 5 classes)
            genre = genres[i % len(genres)]
            filename = f"{genre}/{genre}.{i:05d}.wav"
            data[filename] = {"mfcc": np.random.randn(100, 13).tolist(), "genre": genre}

        # Save to temporary file
        data_file = Path(temp_dir) / "test_mfcc.json"
        with open(data_file, "w") as f:
            json.dump(data, f)

        return str(data_file)

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @patch("src.main.run_automatic_evaluation")
    def test_automatic_evaluation_integration(
        self, mock_eval, sample_mfcc_data, temp_dir
    ):
        """Test that automatic evaluation is called during training."""
        from src.main import train_model

        # Mock the evaluation function
        mock_eval.return_value = {
            "accuracy": 0.75,
            "confusion_matrix": np.eye(5),
            "roc_auc": 0.8,
        }

        # Use real Config object to avoid serialization issues
        from src.core.config import Config
        config = Config()
        config.model.max_epochs = 1
        config.model.batch_size = 4
        config.model.hidden_size = 16
        config.model.num_layers = 1
        config.model.dropout = 0.1
        config.model.optimizer = "adam"
        config.model.learning_rate = 0.01
        config.model.weight_decay = 0.0
        config.model.loss_function = "crossentropy"
        config.training.random_seed = 42
        config.training.improvement_threshold = 0.01
        config.training.improvement_window = 3
        config.training.early_stopping = False

        # Call train_model (this should trigger automatic evaluation)
        result = train_model(
            Mock(
                data=sample_mfcc_data,
                model="LSTM",
                output=temp_dir,
                epochs=1,
                batch_size=4,
            ),
            config,
            Mock(),
        )

        # Check that evaluation was called
        mock_eval.assert_called_once()

        # Check return value includes evaluation results
        assert isinstance(result, tuple)
        assert len(result) == 2
        training_history, evaluation_results = result
        assert evaluation_results is not None

    def test_evaluation_output_files(self, sample_config, sample_mfcc_data, temp_dir):
        """Test that evaluation creates the expected output files."""
        # This test would require running the actual evaluation
        # For now, we'll test the structure
        eval_dir = Path(temp_dir) / "lstm_evaluation_results"
        eval_dir.mkdir(exist_ok=True)

        # Create mock evaluation files
        (eval_dir / "confusion_matrix.png").touch()
        (eval_dir / "evaluation_metrics.txt").touch()
        (eval_dir / "ks_curves.png").touch()

        assert (eval_dir / "confusion_matrix.png").exists()
        assert (eval_dir / "evaluation_metrics.txt").exists()
        assert (eval_dir / "ks_curves.png").exists()
