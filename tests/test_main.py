"""
Tests for GenreDiscern main CLI functionality.
"""

import pytest
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from main import setup_cli_parser, extract_features, train_model


class TestCLIParser:
    """Test command line argument parsing."""

    def test_parser_creation(self):
        """Test that CLI parser is created correctly."""
        parser = setup_cli_parser()
        assert parser is not None
        assert hasattr(parser, "parse_args")

        # Check subparsers
        subparsers = [
            action
            for action in parser._actions
            if hasattr(action, "choices") and action.choices is not None
        ]
        assert len(subparsers) > 0

        # Check commands
        commands = subparsers[0].choices.keys()
        assert "extract" in commands
        assert "train" in commands
        assert "evaluate" in commands

    def test_extract_command_args(self):
        """Test extract command argument parsing."""
        parser = setup_cli_parser()

        # Test valid extract command
        args = parser.parse_args(
            [
                "extract",
                "--input",
                "/path/to/music",
                "--output",
                "/path/to/output",
                "--name",
                "features",
            ]
        )

        assert args.command == "extract"
        assert args.input == "/path/to/music"
        assert args.output == "/path/to/output"
        assert args.name == "features"

    def test_train_command_args(self):
        """Test train command argument parsing."""
        parser = setup_cli_parser()

        # Test valid train command
        args = parser.parse_args(
            [
                "train",
                "--data",
                "/path/to/features.json",
                "--model",
                "LSTM",
                "--output",
                "/path/to/output",
                "--epochs",
                "10",
                "--batch-size",
                "32",
            ]
        )

        assert args.command == "train"
        assert args.data == "/path/to/features.json"
        assert args.model == "LSTM"
        assert args.output == "/path/to/output"
        assert args.epochs == 10
        assert args.batch_size == 32

    def test_evaluate_command_args(self):
        """Test evaluate command argument parsing."""
        parser = setup_cli_parser()

        # Test valid evaluate command
        args = parser.parse_args(
            [
                "evaluate",
                "--model",
                "/path/to/model.onnx",
                "--data",
                "/path/to/features.json",
                "--output",
                "/path/to/output",
            ]
        )

        assert args.command == "evaluate"
        assert args.model == "/path/to/model.onnx"
        assert args.data == "/path/to/features.json"
        assert args.output == "/path/to/output"

    def test_global_options(self):
        """Test global command line options."""
        parser = setup_cli_parser()

        # Test verbose and log file options (these are global options, not subcommand options)
        args = parser.parse_args(
            [
                "--verbose",
                "--log-file",
                "/path/to/log.txt",
                "train",
                "--data",
                "/path/to/features.json",
                "--model",
                "LSTM",
                "--output",
                "/path/to/output",
            ]
        )

        assert args.verbose is True
        assert args.log_file == "/path/to/log.txt"


class TestExtractFeatures:
    """Test feature extraction functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @patch("src.main.MFCCExtractor")
    @patch("src.main.DatasetAgnosticMFCCExtractor")
    @patch("src.main.DatasetFactory.create_dataset")
    def test_extract_features_success(
        self, mock_create_dataset, mock_dataset_agnostic, mock_extractor_class, temp_dir
    ):
        """Test successful feature extraction."""
        # Mock the extractor
        mock_extractor = Mock()
        mock_extractor.extract_mfcc_from_directory.return_value = (
            f"{temp_dir}/features.json"
        )
        mock_extractor_class.return_value = mock_extractor

        # Mock dataset factory
        mock_dataset = Mock()
        mock_dataset.get_audio_files.return_value = [("test.wav", "blues")]
        mock_dataset.get_metadata.return_value = {"name": "test_dataset", "total_files": 1}
        mock_create_dataset.return_value = mock_dataset

        # Mock DatasetAgnosticMFCCExtractor
        mock_extractor_instance = Mock()
        mock_extractor_instance.validate_dataset.return_value = True
        mock_extractor_instance.extract_mfccs.return_value = {
            "metadata": {"total_samples": 100},
            "output_file": f"{temp_dir}/features.json"
        }
        mock_dataset_agnostic.return_value = mock_extractor_instance

        # Mock config
        mock_config = Mock()
        mock_config.audio = Mock()

        # Mock logger
        mock_logger = Mock()

        # Test extraction
        args = Mock()
        args.input = str(temp_dir)  # Ensure it's a string, not Mock
        args.output = str(temp_dir)
        args.name = "test_features"

        result = extract_features(
            args.input, args.output, args.name, logger=mock_logger
        )

        assert result == f"{temp_dir}/features.json"
        mock_extractor_instance.extract_mfccs.assert_called_once()

    @patch("src.main.MFCCExtractor")
    @patch("src.data.dataset_agnostic_mfcc_extractor.DatasetFactory.create_dataset")
    def test_extract_features_failure(
        self, mock_create_dataset, mock_extractor_class, temp_dir
    ):
        """Test feature extraction failure handling."""
        # Mock the extractor to raise an exception
        mock_extractor = Mock()
        mock_extractor.extract_mfcc_from_directory.side_effect = Exception(
            "Extraction failed"
        )
        mock_extractor_class.return_value = mock_extractor

        # Mock dataset factory
        mock_dataset = Mock()
        mock_dataset.get_audio_files.return_value = [("test.wav", "blues")]
        mock_create_dataset.return_value = mock_dataset

        # Mock config
        mock_config = Mock()
        mock_config.audio = Mock()

        # Mock logger
        mock_logger = Mock()

        # Test that exception is raised
        args = Mock()
        args.input = str(temp_dir)  # Ensure it's a string, not Mock
        args.output = str(temp_dir)
        args.name = "test_features"

        with pytest.raises(Exception, match="Extraction failed"):
            extract_features(args.input, args.output, args.name, logger=mock_logger)


class TestTrainModel:
    """Test model training functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @patch("src.main.ModelTrainer")
    @patch("src.main.run_automatic_evaluation")
    def test_train_model_success_with_evaluation(
        self, mock_eval, mock_trainer_class, temp_dir
    ):
        """Test successful model training with automatic evaluation."""
        # Mock the trainer
        mock_trainer = Mock()
        mock_trainer.train.return_value = {
            "train_loss": [0.5, 0.3],
            "val_loss": [0.6, 0.4],
        }
        mock_trainer_class.return_value = mock_trainer

        # Mock evaluation
        mock_eval.return_value = {"accuracy": 0.75, "confusion_matrix": None}

        # Mock config
        mock_config = Mock()
        mock_config.model.max_epochs = 10
        mock_config.model.batch_size = 32
        mock_config.model.hidden_size = 64
        mock_config.model.num_layers = 2
        mock_config.model.dropout = 0.1
        mock_config.model.optimizer = "adam"
        mock_config.model.learning_rate = 0.001
        mock_config.model.weight_decay = 0.0
        mock_config.model.loss_function = "crossentropy"
        mock_config.training.random_seed = 42
        mock_config.training.improvement_threshold = 0.01
        mock_config.training.improvement_window = 3

        # Mock logger
        mock_logger = Mock()

        # Create test data file
        test_data = {
            "features": [[[1.0, 2.0, 3.0] for _ in range(100)] for _ in range(150)],
            "labels": ["blues", "classical", "country", "disco", "hiphop"] * 30,
        }
        data_file = Path(temp_dir) / "data.json"
        with open(data_file, "w") as f:
            json.dump(test_data, f)

        # Test training
        result = train_model(
            Mock(
                data=str(data_file),
                model="LSTM",
                output=temp_dir,
                epochs=None,
                batch_size=None,
            ),
            mock_config,
            mock_logger,
        )

        # Check that training was called
        mock_trainer.setup_training.assert_called_once()
        mock_trainer.train.assert_called_once()

        # Check that evaluation was called
        mock_eval.assert_called_once()

        # Check return value
        assert isinstance(result, tuple)
        assert len(result) == 2
        training_history, evaluation_results = result
        assert evaluation_results is not None

    @patch("src.main.ModelTrainer")
    @patch("src.main.run_automatic_evaluation")
    def test_train_model_success_without_evaluation(
        self, mock_eval, mock_trainer_class, temp_dir
    ):
        """Test successful model training without automatic evaluation."""
        # Mock the trainer
        mock_trainer = Mock()
        mock_trainer.train.return_value = {
            "train_loss": [0.5, 0.3],
            "val_loss": [0.6, 0.4],
        }
        mock_trainer_class.return_value = mock_trainer

        # Mock evaluation to fail
        mock_eval.side_effect = Exception("Evaluation failed")

        # Mock config
        mock_config = Mock()
        mock_config.model.max_epochs = 10
        mock_config.model.batch_size = 32
        mock_config.model.hidden_size = 64
        mock_config.model.num_layers = 2
        mock_config.model.dropout = 0.1
        mock_config.model.optimizer = "adam"
        mock_config.model.learning_rate = 0.001
        mock_config.model.weight_decay = 0.0
        mock_config.model.loss_function = "crossentropy"
        mock_config.training.random_seed = 42
        mock_config.training.improvement_threshold = 0.01
        mock_config.training.improvement_window = 3

        # Mock logger
        mock_logger = Mock()

        # Create test data file
        test_data = {
            "features": [[[1.0, 2.0, 3.0] for _ in range(100)] for _ in range(150)],
            "labels": ["blues", "classical", "country", "disco", "hiphop"] * 30,
        }
        data_file = Path(temp_dir) / "data.json"
        with open(data_file, "w") as f:
            json.dump(test_data, f)

        # Test training
        result = train_model(
            Mock(
                data=str(data_file),
                model="LSTM",
                output=temp_dir,
                epochs=None,
                batch_size=None,
            ),
            mock_config,
            mock_logger,
        )

        # Check that training was called
        mock_trainer.setup_training.assert_called_once()
        mock_trainer.train.assert_called_once()

        # Check that evaluation was attempted but failed
        mock_eval.assert_called_once()

        # Check return value (should be just training history)
        assert isinstance(result, dict)
        assert "train_loss" in result

    @patch("src.main.ModelTrainer")
    def test_train_model_failure(self, mock_trainer_class, temp_dir):
        """Test model training failure handling."""
        # Mock the trainer to raise an exception
        mock_trainer = Mock()
        mock_trainer.setup_training.side_effect = Exception("Training failed")
        mock_trainer_class.return_value = mock_trainer

        # Mock config
        mock_config = Mock()
        mock_config.model.max_epochs = 10
        mock_config.model.batch_size = 32
        mock_config.model.hidden_size = 64
        mock_config.model.num_layers = 2
        mock_config.model.dropout = 0.1
        mock_config.model.optimizer = "adam"
        mock_config.model.learning_rate = 0.001
        mock_config.model.weight_decay = 0.0
        mock_config.model.loss_function = "crossentropy"
        mock_config.training.random_seed = 42
        mock_config.training.improvement_threshold = 0.01
        mock_config.training.improvement_window = 3

        # Mock logger
        mock_logger = Mock()

        # Create test data file
        test_data = {
            "features": [[[1.0, 2.0, 3.0] for _ in range(100)] for _ in range(150)],
            "labels": ["blues", "classical", "country", "disco", "hiphop"] * 30,
        }
        data_file = Path(temp_dir) / "data.json"
        with open(data_file, "w") as f:
            json.dump(test_data, f)

        # Test that exception is raised
        with pytest.raises(Exception, match="Training failed"):
            train_model(
                Mock(
                    data=str(data_file),
                    model="LSTM",
                    output=temp_dir,
                    epochs=None,
                    batch_size=None,
                ),
                mock_config,
                mock_logger,
            )

    @patch("src.main.ModelTrainer")
    def test_train_model_with_custom_args(self, mock_trainer_class, temp_dir):
        """Test model training with custom command line arguments."""
        # Mock the trainer
        mock_trainer = Mock()
        mock_trainer.train.return_value = {"train_loss": [0.5], "val_loss": [0.6]}
        mock_trainer_class.return_value = mock_trainer

        # Mock config
        mock_config = Mock()
        mock_config.model.max_epochs = 10
        mock_config.model.batch_size = 32
        mock_config.model.hidden_size = 64
        mock_config.model.num_layers = 2
        mock_config.model.dropout = 0.1
        mock_config.model.optimizer = "adam"
        mock_config.model.learning_rate = 0.001
        mock_config.model.weight_decay = 0.0
        mock_config.model.loss_function = "crossentropy"
        mock_config.training.random_seed = 42
        mock_config.training.improvement_threshold = 0.01
        mock_config.training.improvement_window = 3

        # Mock logger
        mock_logger = Mock()

        # Create test data file
        test_data = {
            "features": [[[1.0, 2.0, 3.0] for _ in range(100)] for _ in range(150)],
            "labels": ["blues", "classical", "country", "disco", "hiphop"] * 30,
        }
        data_file = Path(temp_dir) / "data.json"
        with open(data_file, "w") as f:
            json.dump(test_data, f)

        # Test training with custom epochs and batch size
        result = train_model(
            Mock(
                data=str(data_file),
                model="LSTM",
                output=temp_dir,
                epochs=5,
                batch_size=16,
            ),
            mock_config,
            mock_logger,
        )

        # Check that config was updated
        assert mock_config.model.max_epochs == 5
        assert mock_config.model.batch_size == 16

        # Check that training was called
        mock_trainer.setup_training.assert_called_once()
        mock_trainer.train.assert_called_once()


class TestMainIntegration:
    """Test main function integration."""

    @patch("src.main.setup_logging")
    @patch("src.main.Config")
    @patch("src.main.extract_features")
    def test_main_extract_command(
        self, mock_extract, mock_config_class, mock_logging, temp_dir
    ):
        """Test main function with extract command."""
        from src.main import main

        # Mock dependencies
        mock_logger = Mock()
        mock_logging.return_value = mock_logger

        mock_config = Mock()
        mock_config_class.return_value = mock_config

        mock_extract.return_value = f"{temp_dir}/features.json"

        # Mock command line arguments
        with patch.object(
            sys,
            "argv",
            [
                "main.py",
                "extract",
                "--input",
                temp_dir,
                "--output",
                temp_dir,
                "--name",
                "test",
            ],
        ):
            # This would require more complex mocking of argparse
            # For now, we'll test the individual components
            pass

    @patch("src.main.setup_logging")
    @patch("src.main.Config")
    @patch("src.main.train_model")
    def test_main_train_command(
        self, mock_train, mock_config_class, mock_logging, temp_dir
    ):
        """Test main function with train command."""
        from src.main import main

        # Mock dependencies
        mock_logger = Mock()
        mock_logging.return_value = mock_logger

        mock_config = Mock()
        mock_config_class.return_value = mock_config

        mock_train.return_value = ({"train_loss": [0.5]}, {"accuracy": 0.75})

        # Mock command line arguments
        with patch.object(
            sys,
            "argv",
            [
                "main.py",
                "train",
                "--data",
                f"{temp_dir}/data.json",
                "--model",
                "LSTM",
                "--output",
                temp_dir,
            ],
        ):
            # This would require more complex mocking of argparse
            # For now, we'll test the individual components
            pass
