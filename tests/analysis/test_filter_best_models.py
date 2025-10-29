#!/usr/bin/env python3
"""
Unit tests for filter_best_models.py.
Tests the best models filtering functionality.
"""

import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analysis.filter_best_models import filter_overfitting_analysis, get_best_models


class TestFilterBestModels(unittest.TestCase):
    """Test cases for filter_best_models functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_analysis_dir = os.path.join(self.temp_dir, "outputs", "analysis")
        os.makedirs(self.test_analysis_dir, exist_ok=True)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_mock_results_summary(self):
        """Create a mock results_summary.csv."""
        data = {
            "run_dir": [
                "gtzan-fc-1",
                "gtzan-fc-2",
                "gtzan-cnn-1",
                "fma-lstm-1",
            ],
            "dataset": ["GTZAN", "GTZAN", "GTZAN", "FMA"],
            "model": ["FC", "FC", "CNN", "LSTM"],
            "train_acc": [0.95, 0.92, 0.93, 0.94],
            "val_acc": [0.90, 0.88, 0.89, 0.91],
            "test_acc": [0.88, 0.85, 0.87, 0.89],
            "roc_auc": [0.92, 0.90, 0.91, 0.93],
        }
        df = pd.DataFrame(data)
        csv_path = os.path.join(self.test_analysis_dir, "results_summary.csv")
        df.to_csv(csv_path, index=False)
        return df

    def create_mock_overfitting_analysis(self):
        """Create a mock overfitting_analysis.csv."""
        data = {
            "model": ["FC", "FC", "CNN", "LSTM"],
            "dataset": ["GTZAN", "GTZAN", "GTZAN", "FMA"],
            "run_dir": ["gtzan-fc-1", "gtzan-fc-2", "gtzan-cnn-1", "fma-lstm-1"],
            "train_acc": [0.95, 0.92, 0.93, 0.94],
            "val_acc": [0.90, 0.88, 0.89, 0.91],
            "test_acc": [0.88, 0.85, 0.87, 0.89],
            "overfitting": [0.05, 0.04, 0.04, 0.03],
            "epochs": [50, 50, 50, 50],
        }
        df = pd.DataFrame(data)
        csv_path = os.path.join(self.test_analysis_dir, "overfitting_analysis.csv")
        df.to_csv(csv_path, index=False)
        return df

    def test_get_best_models_basic(self):
        """Test getting best models with basic data."""
        # Change to temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            self.create_mock_results_summary()

            best_mapping, best_df = get_best_models()

            # Should have 3 unique model-dataset combinations
            self.assertEqual(len(best_mapping), 3)
            self.assertIn(("FC", "GTZAN"), best_mapping)
            self.assertIn(("CNN", "GTZAN"), best_mapping)
            self.assertIn(("LSTM", "FMA"), best_mapping)

            # FC-GTZAN should have highest test_acc (0.88)
            self.assertEqual(best_mapping[("FC", "GTZAN")], "gtzan-fc-1")
        finally:
            os.chdir(original_cwd)

    def test_get_best_models_with_nan(self):
        """Test getting best models when some rows have NaN test_acc."""
        # Change to temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            data = {
                "run_dir": ["gtzan-fc-1", "gtzan-cnn-1"],
                "dataset": ["GTZAN", "GTZAN"],
                "model": ["FC", "CNN"],
                "train_acc": [0.95, 0.93],
                "val_acc": [0.90, 0.89],
                "test_acc": [0.88, float("nan")],  # CNN has NaN
                "roc_auc": [0.92, 0.91],
            }
            df = pd.DataFrame(data)
            csv_path = os.path.join(self.test_analysis_dir, "results_summary.csv")
            df.to_csv(csv_path, index=False)

            best_mapping, best_df = get_best_models()

            # Should only have FC-GTZAN (CNN filtered out due to NaN)
            self.assertEqual(len(best_mapping), 1)
            self.assertIn(("FC", "GTZAN"), best_mapping)
        finally:
            os.chdir(original_cwd)

    def test_filter_overfitting_analysis_basic(self):
        """Test filtering overfitting analysis with basic data."""
        # Change to temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            self.create_mock_results_summary()
            self.create_mock_overfitting_analysis()

            fma_table, gtzan_table, best_df = filter_overfitting_analysis()

            # Should have filtered results
            self.assertIsNotNone(fma_table)
            self.assertIsNotNone(gtzan_table)
            self.assertIsNotNone(best_df)

            # GTZAN table should have 2 models (FC, CNN)
            self.assertEqual(len(gtzan_table), 2)

            # FMA table should have 1 model (LSTM)
            self.assertEqual(len(fma_table), 1)
        finally:
            os.chdir(original_cwd)

    def test_filter_overfitting_analysis_no_matches(self):
        """Test filtering when there are no matching models."""
        # Change to temp directory
        original_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            # Create results with only one model (SVM)
            data1 = {
                "run_dir": ["gtzan-svm-1"],
                "dataset": ["GTZAN"],
                "model": ["SVM"],
                "train_acc": [0.95],
                "val_acc": [0.90],
                "test_acc": [0.88],
                "roc_auc": [0.92],
            }
            df1 = pd.DataFrame(data1)
            csv_path1 = os.path.join(self.test_analysis_dir, "results_summary.csv")
            df1.to_csv(csv_path1, index=False)

            # Overfitting analysis has a different model (FC) that doesn't match
            data2 = {
                "model": ["FC"],
                "dataset": ["GTZAN"],
                "run_dir": ["gtzan-fc-1"],
                "train_acc": [0.95],
                "val_acc": [0.90],
                "test_acc": [0.88],
                "overfitting": [0.05],
                "epochs": [50],
            }
            df2 = pd.DataFrame(data2)
            csv_path2 = os.path.join(self.test_analysis_dir, "overfitting_analysis.csv")
            df2.to_csv(csv_path2, index=False)

            fma_table, gtzan_table, best_df = filter_overfitting_analysis()

            # GTZAN table should be empty (no FC in results_summary)
            # But it may contain headers, so check length
            self.assertEqual(len(gtzan_table), 0)
            # FMA table should be empty
            self.assertEqual(len(fma_table), 0)
            # best_df should only contain SVM
            self.assertEqual(len(best_df), 1)
            self.assertEqual(best_df.iloc[0]["model"], "SVM")
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
