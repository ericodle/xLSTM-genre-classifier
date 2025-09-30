"""
Tests for grid search functionality.
"""

import pytest
import json
import tempfile
import os
import sys
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from training.grid_search import GridSearchTrainer
from core.config import Config


class TestGridSearchTrainer:
    """Test the GridSearchTrainer class."""

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        config = Mock(spec=Config)
        config.model = Mock()
        config.training = Mock()
        return config

    @pytest.fixture
    def sample_param_grid(self):
        """Sample parameter grid for testing."""
        return {"hidden_size": [32, 64], "num_layers": [1, 2], "dropout": [0.1, 0.2]}

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield tmp_dir

    def test_grid_search_trainer_initialization(self, sample_config):
        """Test GridSearchTrainer initialization."""
        trainer = GridSearchTrainer(sample_config)
        assert trainer.config == sample_config
        assert trainer.results == []

    def test_grid_search_trainer_default_config(self):
        """Test GridSearchTrainer with default config."""
        trainer = GridSearchTrainer()
        assert trainer.config is not None
        assert trainer.logger is not None

    @patch("training.grid_search.ModelTrainer")
    @patch("training.grid_search.ensure_directory")
    def test_run_grid_search_basic(
        self,
        mock_ensure_dir,
        mock_trainer_class,
        sample_config,
        sample_param_grid,
        temp_dir,
    ):
        """Test basic grid search execution."""
        # Mock the trainer
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer

        # Mock training setup and execution
        mock_trainer.setup_training.return_value = None
        mock_trainer.train.return_value = {
            "train_losses": [0.5, 0.3],
            "val_losses": [0.6, 0.4],
            "train_accuracies": [0.8, 0.9],
            "val_accuracies": [0.7, 0.85],
        }

        # Create trainer and run grid search
        grid_trainer = GridSearchTrainer(sample_config)
        results = grid_trainer.run_grid_search(
            data_path="test_data.json",
            model_type="GRU",
            base_output_dir=temp_dir,
            param_grid=sample_param_grid,
        )

        # Verify results
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 8  # 2^3 = 8 combinations
        assert all(results["status"] == "completed")

        # Verify trainer was called for each combination
        assert mock_trainer_class.call_count == 8
        assert mock_trainer.setup_training.call_count == 8
        assert mock_trainer.train.call_count == 8

    @patch("training.grid_search.ModelTrainer")
    @patch("training.grid_search.ensure_directory")
    def test_run_grid_search_with_failures(
        self,
        mock_ensure_dir,
        mock_trainer_class,
        sample_config,
        sample_param_grid,
        temp_dir,
    ):
        """Test grid search with some training failures."""
        # Mock the trainer
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer

        # First combination succeeds, second fails
        def mock_train():
            if mock_trainer.setup_training.call_count == 1:
                return {
                    "train_losses": [0.5, 0.3],
                    "val_losses": [0.6, 0.4],
                    "train_accuracies": [0.8, 0.9],
                    "val_accuracies": [0.7, 0.85],
                }
            else:
                raise Exception("Training failed")

        mock_trainer.setup_training.return_value = None
        mock_trainer.train.side_effect = mock_train

        # Create trainer and run grid search
        grid_trainer = GridSearchTrainer(sample_config)
        results = grid_trainer.run_grid_search(
            data_path="test_data.json",
            model_type="GRU",
            base_output_dir=temp_dir,
            param_grid=sample_param_grid,
        )

        # Verify results include both successes and failures
        assert len(results) == 8
        successful = results[results["status"] == "completed"]
        failed = results[results["status"] == "failed"]

        assert len(successful) == 1
        assert len(failed) == 7

    def test_create_output_dir(self, sample_config, temp_dir):
        """Test output directory creation."""
        trainer = GridSearchTrainer(sample_config)
        params = {"hidden_size": 64, "dropout": 0.2}

        output_dir = trainer._create_output_dir(temp_dir, params)

        assert "hidden_size64" in output_dir
        assert "dropout0p200" in output_dir
        assert os.path.exists(output_dir)

    def test_save_results(self, sample_config, temp_dir):
        """Test results saving functionality."""
        trainer = GridSearchTrainer(sample_config)
        trainer.results = [
            {"param1": "value1", "accuracy": 0.8},
            {"param2": "value2", "accuracy": 0.9},
        ]

        trainer._save_results(temp_dir)

        # Check that files were created
        assert os.path.exists(os.path.join(temp_dir, "grid_search_results.json"))
        assert os.path.exists(os.path.join(temp_dir, "grid_search_results.csv"))

    def test_generate_summary(self, sample_config):
        """Test summary generation."""
        trainer = GridSearchTrainer(sample_config)
        trainer.results = [
            {"status": "completed", "best_val_acc": 0.8, "best_val_loss": 0.3},
            {"status": "completed", "best_val_acc": 0.9, "best_val_loss": 0.2},
            {"status": "failed", "error": "Training error"},
        ]

        summary = trainer._generate_summary()

        assert "total_combinations" in summary
        assert "successful_runs" in summary
        assert "failed_runs" in summary
        assert summary["total_combinations"] == 3
        assert summary["successful_runs"] == 2
        assert summary["failed_runs"] == 1
        assert "best_accuracy" in summary
        assert "best_loss" in summary
        assert "statistics" in summary


class TestRunGridSearchScript:
    """Test the run_grid_search.py script functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield tmp_dir

    @pytest.fixture
    def sample_mfcc_data(self):
        """Create sample MFCC data file."""
        data = {
            "features": [[1.0, 2.0, 3.0] for _ in range(100)],
            "labels": ["genre1", "genre2"] * 50,
            "metadata": {"sample_rate": 22050, "n_mfcc": 13},
        }
        return data

    def test_load_param_grid_defaults(self):
        """Test loading default parameter grids."""
        # Import the function from the script
        import importlib
        import run_grid_search

        importlib.reload(run_grid_search)  # Ensure fresh import

        # Test default grids for different models
        gru_grid = run_grid_search.load_param_grid()
        assert "GRU" in gru_grid
        assert "hidden_size" in gru_grid["GRU"]
        assert "num_layers" in gru_grid["GRU"]

        cnn_grid = run_grid_search.load_param_grid()
        assert "CNN" in cnn_grid
        assert "num_filters" in cnn_grid["CNN"]
        assert "kernel_size" in cnn_grid["CNN"]

    def test_load_param_grid_custom_file(self, temp_dir):
        """Test loading custom parameter grid from file."""
        # Create custom parameter file
        custom_params = {"hidden_size": [64, 128], "dropout": [0.1, 0.3]}

        param_file = os.path.join(temp_dir, "custom_params.json")
        with open(param_file, "w") as f:
            json.dump(custom_params, f)

        # Import and test
        import importlib
        import run_grid_search

        importlib.reload(run_grid_search)

        loaded_params = run_grid_search.load_param_grid(param_file)
        assert loaded_params == custom_params

    def test_main_function_basic(self, temp_dir, sample_mfcc_data):
        """Test main function with basic execution."""
        # Create sample MFCC file
        mfcc_file = os.path.join(temp_dir, "test_mfcc.json")
        with open(mfcc_file, "w") as f:
            json.dump(sample_mfcc_data, f)

        # Test that the script can be imported and has the expected functions
        import run_grid_search

        # Check that the script has the expected functions
        assert hasattr(run_grid_search, "main")
        assert hasattr(run_grid_search, "load_param_grid")
        assert hasattr(run_grid_search, "setup_parser")

        # Test that load_param_grid works
        gru_grid = run_grid_search.load_param_grid()
        assert "GRU" in gru_grid
        assert "hidden_size" in gru_grid["GRU"]

    def test_dry_run_functionality(self, temp_dir, sample_mfcc_data):
        """Test dry-run functionality."""
        # Create sample MFCC file
        mfcc_file = os.path.join(temp_dir, "test_mfcc.json")
        with open(mfcc_file, "w") as f:
            json.dump(sample_mfcc_data, f)

        # Import the script
        import importlib
        import run_grid_search

        importlib.reload(run_grid_search)

        # Mock command line arguments for dry run
        with patch(
            "sys.argv",
            [
                "run_grid_search.py",
                "--model",
                "GRU",
                "--data",
                mfcc_file,
                "--output",
                temp_dir,
                "--dry-run",
            ],
        ):
            # Capture stdout to check output
            from io import StringIO

            with patch("sys.stdout", StringIO()) as mock_stdout:
                run_grid_search.main()
                output = mock_stdout.getvalue()

                # Check that dry run output is present
                assert "DRY RUN" in output
                assert "Total combinations:" in output
                assert "Total training runs:" in output

    def test_parameter_validation(self, temp_dir):
        """Test parameter validation."""
        import importlib
        import run_grid_search

        importlib.reload(run_grid_search)

        # Test with invalid model type
        with patch(
            "sys.argv",
            [
                "run_grid_search.py",
                "--model",
                "INVALID",
                "--data",
                "nonexistent.json",
                "--output",
                temp_dir,
            ],
        ):
            with pytest.raises(SystemExit):
                run_grid_search.main()

        # Test with nonexistent data file
        with patch(
            "sys.argv",
            [
                "run_grid_search.py",
                "--model",
                "GRU",
                "--data",
                "nonexistent.json",
                "--output",
                temp_dir,
            ],
        ):
            with pytest.raises(SystemExit):
                run_grid_search.main()


class TestGridSearchIntegration:
    """Integration tests for grid search with actual training components."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield tmp_dir

    @patch("training.grid_search.ModelTrainer")
    def test_grid_search_with_model_trainer_integration(
        self, mock_trainer_class, temp_dir
    ):
        """Test grid search integration with ModelTrainer."""
        # Mock the trainer to simulate actual training
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer

        # Simulate successful training
        mock_trainer.setup_training.return_value = None
        mock_trainer.train.return_value = {
            "train_losses": [0.8, 0.6, 0.4],
            "val_losses": [0.9, 0.7, 0.5],
            "train_accuracies": [0.6, 0.8, 0.9],
            "val_accuracies": [0.5, 0.7, 0.85],
        }

        # Create small parameter grid for testing
        param_grid = {"hidden_size": [32], "dropout": [0.1]}

        # Run grid search
        grid_trainer = GridSearchTrainer()
        results = grid_trainer.run_grid_search(
            data_path="test_data.json",
            model_type="GRU",
            base_output_dir=temp_dir,
            param_grid=param_grid,
        )

        # Verify integration worked
        assert len(results) == 1
        assert results.iloc[0]["status"] == "completed"
        assert results.iloc[0]["best_val_acc"] == 0.85
        assert results.iloc[0]["hidden_size"] == 32
        assert results.iloc[0]["dropout"] == 0.1
