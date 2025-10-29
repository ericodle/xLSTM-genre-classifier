#!/usr/bin/env python3
"""
Unit tests for overfitting_analysis.py.
Tests the overfitting analysis functionality.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from src.analysis.overfitting_analysis import find_best_models, load_training_history


class TestOverfittingAnalysis(unittest.TestCase):
    """Test cases for overfitting analysis functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_outputs_dir = os.path.join(self.temp_dir, "outputs")
        os.makedirs(self.test_outputs_dir, exist_ok=True)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_mock_evaluation_dir(self, run_dir):
        """Create a mock evaluation directory with results.json."""
        eval_dir = os.path.join(self.test_outputs_dir, run_dir, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        json_path = os.path.join(eval_dir, "evaluation_results.json")
        with open(json_path, "w") as f:
            json.dump({"accuracy": 0.85}, f)
        return eval_dir

    def create_mock_metadata(self, run_dir, train_acc, val_acc):
        """Create a mock model_metadata.json file."""
        dir_path = os.path.join(self.test_outputs_dir, run_dir)
        os.makedirs(dir_path, exist_ok=True)
        json_path = os.path.join(dir_path, "model_metadata.json")
        data = {
            "model_type": "FC_model",
            "training_history": {
                "train_acc": train_acc,
                "val_acc": val_acc,
            },
        }
        with open(json_path, "w") as f:
            json.dump(data, f)
        return json_path

    def test_find_best_models_empty_directory(self):
        """Test finding best models in empty directory."""
        best_models = find_best_models(self.test_outputs_dir)
        self.assertEqual(len(best_models), 0)

    def test_find_best_models_single_model(self):
        """Test finding best model with single training run."""
        # Create mock evaluation
        run_dir = "gtzan-fc"
        self.create_mock_evaluation_dir(run_dir)

        # Create mock metadata
        train_acc = [0.9, 0.92, 0.94, 0.95]
        val_acc = [0.85, 0.88, 0.90, 0.91]
        self.create_mock_metadata(run_dir, train_acc, val_acc)

        best_models = find_best_models(self.test_outputs_dir)
        self.assertEqual(len(best_models), 1)
        self.assertIn(("FC", "GTZAN"), best_models)
        self.assertEqual(best_models[("FC", "GTZAN")]["test_acc"], 0.85)

    def test_find_best_models_multiple_runs_same_model(self):
        """Test finding best model with multiple runs of same model type."""
        # Create first run
        run_dir1 = "gtzan-fc"
        self.create_mock_evaluation_dir(run_dir1)
        self.create_mock_metadata(run_dir1, [0.9, 0.92], [0.85, 0.88])

        # Create second run with better test accuracy
        run_dir2 = "gtzan-fc2"
        eval_dir2 = os.path.join(self.test_outputs_dir, run_dir2, "evaluation")
        os.makedirs(eval_dir2, exist_ok=True)
        json_path2 = os.path.join(eval_dir2, "evaluation_results.json")
        with open(json_path2, "w") as f:
            json.dump({"accuracy": 0.95}, f)
        self.create_mock_metadata(run_dir2, [0.92, 0.95], [0.88, 0.93])

        best_models = find_best_models(self.test_outputs_dir)
        self.assertEqual(len(best_models), 1)
        self.assertEqual(best_models[("FC", "GTZAN")]["test_acc"], 0.95)
        self.assertEqual(best_models[("FC", "GTZAN")]["run_dir"], run_dir2)

    def test_load_training_history_valid_data(self):
        """Test loading training history from valid metadata."""
        # Create mock metadata
        train_acc = [0.9, 0.92, 0.94, 0.95]
        val_acc = [0.85, 0.88, 0.90, 0.91]
        json_path = self.create_mock_metadata("gtzan-fc", train_acc, val_acc)

        result = load_training_history(json_path)
        self.assertIsNotNone(result)
        last_train, last_val, epochs = result
        self.assertEqual(last_train, 0.95)
        self.assertEqual(last_val, 0.91)
        self.assertEqual(epochs, 4)

    def test_load_training_history_empty_data(self):
        """Test loading training history from empty arrays."""
        json_path = self.create_mock_metadata("gtzan-fc", [], [])

        result = load_training_history(json_path)
        self.assertIsNotNone(result)
        train, val, epochs = result
        # Should return None for empty arrays
        self.assertIsNone(train)
        self.assertIsNone(val)
        self.assertIsNone(epochs)

    def test_load_training_history_missing_history(self):
        """Test loading training history when history is missing."""
        dir_path = os.path.join(self.test_outputs_dir, "gtzan-fc")
        os.makedirs(dir_path, exist_ok=True)
        json_path = os.path.join(dir_path, "model_metadata.json")
        with open(json_path, "w") as f:
            json.dump({"model_type": "FC_model"}, f)

        result = load_training_history(json_path)
        self.assertIsNotNone(result)
        train, val, epochs = result
        # Should return None values
        self.assertIsNone(train)
        self.assertIsNone(val)
        self.assertIsNone(epochs)

    def test_load_training_history_invalid_json(self):
        """Test loading training history from invalid JSON."""
        dir_path = os.path.join(self.test_outputs_dir, "gtzan-fc")
        os.makedirs(dir_path, exist_ok=True)
        json_path = os.path.join(dir_path, "model_metadata.json")
        with open(json_path, "w") as f:
            f.write("invalid json")

        result = load_training_history(json_path)
        # Should return None values on error
        train, val, epochs = result
        self.assertIsNone(train)
        self.assertIsNone(val)
        self.assertIsNone(epochs)


if __name__ == "__main__":
    unittest.main()
