#!/usr/bin/env python3
"""
Unit tests for analyze_results.py.
Tests the results aggregation functionality.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analysis.analyze_results import collect_results


class TestAnalyzeResults(unittest.TestCase):
    """Test cases for analyze_results functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_outputs_dir = os.path.join(self.temp_dir, "outputs")
        os.makedirs(self.test_outputs_dir, exist_ok=True)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_mock_results_json(self, dir_name, data):
        """Create a mock results.json file."""
        dir_path = os.path.join(self.test_outputs_dir, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        json_path = os.path.join(dir_path, "results.json")
        with open(json_path, "w") as f:
            json.dump(data, f)
        return dir_path

    def test_collect_results_empty_directory(self):
        """Test collecting results from empty directory."""
        df = collect_results(self.test_outputs_dir)
        self.assertTrue(df.empty)

    def test_collect_results_with_neural_network_data(self):
        """Test collecting results from neural network training runs."""
        # Create mock neural network evaluation files
        dir_path = os.path.join(self.test_outputs_dir, "gtzan-fc")
        os.makedirs(dir_path, exist_ok=True)

        # Create evaluation directory with metrics file
        eval_dir = os.path.join(dir_path, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        metrics_file = os.path.join(eval_dir, "evaluation_metrics.txt")
        with open(metrics_file, "w") as f:
            f.write("Accuracy: 0.88\nROC AUC: 0.92\n")

        df = collect_results(self.test_outputs_dir)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["dataset"], "GTZAN")
        self.assertEqual(df.iloc[0]["model"], "FC")
        # For neural networks, train/val are NaN, only test_acc from eval file
        self.assertTrue(pd.isna(df.iloc[0]["train_acc"]))
        self.assertTrue(pd.isna(df.iloc[0]["val_acc"]))
        self.assertEqual(df.iloc[0]["test_acc"], 0.88)

    def test_collect_results_with_svm_data(self):
        """Test collecting results from SVM training runs."""
        # Create mock SVM results
        self.create_mock_results_json(
            "gtzan-svm",
            {
                "train": {"accuracy": 0.92},
                "val": {"accuracy": 0.88},
                "test": {"accuracy": 0.85},
            },
        )

        df = collect_results(self.test_outputs_dir)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["dataset"], "GTZAN")
        self.assertEqual(df.iloc[0]["model"], "SVM")
        self.assertEqual(df.iloc[0]["train_acc"], 0.92)
        self.assertEqual(df.iloc[0]["val_acc"], 0.88)
        self.assertEqual(df.iloc[0]["test_acc"], 0.85)

    def test_collect_results_with_multiple_models(self):
        """Test collecting results from multiple training runs."""
        # Create mock evaluation files for multiple models
        models_data = [
            ("gtzan-fc", 0.88, 0.92),
            ("gtzan-cnn", 0.87, 0.91),
            ("fma-lstm", 0.89, 0.93),
        ]

        for dir_name, test_acc, roc_auc in models_data:
            dir_path = os.path.join(self.test_outputs_dir, dir_name)
            os.makedirs(dir_path, exist_ok=True)
            eval_dir = os.path.join(dir_path, "evaluation")
            os.makedirs(eval_dir, exist_ok=True)
            metrics_file = os.path.join(eval_dir, "evaluation_metrics.txt")
            with open(metrics_file, "w") as f:
                f.write(f"Accuracy: {test_acc}\nROC AUC: {roc_auc}\n")

        df = collect_results(self.test_outputs_dir)
        self.assertEqual(len(df), 3)
        self.assertEqual(set(df["model"]), {"FC", "CNN", "LSTM"})
        self.assertEqual(set(df["dataset"]), {"GTZAN", "FMA"})

    def test_collect_results_with_invalid_json(self):
        """Test collecting results when JSON is invalid."""
        # Create invalid JSON file
        dir_path = os.path.join(self.test_outputs_dir, "gtzan-fc")
        os.makedirs(dir_path, exist_ok=True)
        json_path = os.path.join(dir_path, "results.json")
        with open(json_path, "w") as f:
            f.write("invalid json content")

        df = collect_results(self.test_outputs_dir)
        # Should skip invalid JSON and return empty DataFrame
        self.assertTrue(df.empty)

    def test_collect_results_with_missing_data(self):
        """Test collecting results when some data is missing."""
        # Create evaluation file with only ROC AUC (no accuracy)
        dir_path = os.path.join(self.test_outputs_dir, "gtzan-fc")
        os.makedirs(dir_path, exist_ok=True)
        eval_dir = os.path.join(dir_path, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        metrics_file = os.path.join(eval_dir, "evaluation_metrics.txt")
        with open(metrics_file, "w") as f:
            f.write("ROC AUC: 0.92\n")

        df = collect_results(self.test_outputs_dir)
        self.assertEqual(len(df), 1)
        # test_acc should be NaN when accuracy is missing
        self.assertTrue(pd.isna(df.iloc[0]["test_acc"]))

    def test_collect_results_nonexistent_directory(self):
        """Test collecting results from nonexistent directory."""
        df = collect_results("/nonexistent/path")
        self.assertTrue(df.empty)


if __name__ == "__main__":
    unittest.main()
