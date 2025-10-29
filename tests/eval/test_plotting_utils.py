#!/usr/bin/env python3
"""
Unit tests for plotting_utils.py.
Tests plotting and visualization utilities.
"""

import os
import tempfile
import unittest

import numpy as np
from sklearn.metrics import roc_auc_score

from src.eval.plotting_utils import PlottingUtilities


class TestPlottingUtils(unittest.TestCase):
    """Test cases for PlottingUtilities functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.plotter = PlottingUtilities()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_plot_confusion_matrix_basic(self):
        """Test plotting confusion matrix with basic data."""
        cm = np.array([[5, 2], [1, 7]])
        output_path = os.path.join(self.temp_dir, "confusion.png")

        self.plotter.plot_confusion_matrix(cm, None, output_path)

        # Check that file was created
        self.assertTrue(os.path.exists(output_path))

    def test_plot_confusion_matrix_with_class_names(self):
        """Test plotting confusion matrix with class names."""
        cm = np.array([[5, 2], [1, 7]])
        class_names = ["Class A", "Class B"]
        output_path = os.path.join(self.temp_dir, "confusion_names.png")

        self.plotter.plot_confusion_matrix(cm, class_names, output_path, "Test CM")

        # Check that file was created
        self.assertTrue(os.path.exists(output_path))

    def test_plot_roc_curves_multiclass(self):
        """Test plotting ROC curves for multi-class classification."""
        # Create mock multi-class data
        n_samples = 100
        n_classes = 3
        y_true = np.random.randint(0, n_classes, n_samples)
        y_probs = np.random.rand(n_samples, n_classes)
        y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)

        results = {
            "y_true": y_true,
            "y_probs": y_probs,
            "roc_auc": 0.75,
        }
        class_names = ["Class A", "Class B", "Class C"]
        output_path = os.path.join(self.temp_dir, "roc.png")

        self.plotter.plot_roc_curves(results, class_names, output_path)

        # Check that file was created
        self.assertTrue(os.path.exists(output_path))

    def test_plot_roc_curves_binary(self):
        """Test plotting ROC curves for binary classification."""
        n_samples = 100
        y_true = np.random.randint(0, 2, n_samples)
        y_probs = np.random.rand(n_samples, 2)
        y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)

        results = {
            "y_true": y_true,
            "y_probs": y_probs,
            "roc_auc": roc_auc_score(y_true, y_probs[:, 1]),
        }
        output_path = os.path.join(self.temp_dir, "roc_binary.png")

        self.plotter.plot_roc_curves(results, None, output_path)

        # Check that file was created
        self.assertTrue(os.path.exists(output_path))

    def test_plot_roc_curves_no_data(self):
        """Test plotting ROC curves with no probability data."""
        results = {"roc_auc": 0.75}
        output_path = os.path.join(self.temp_dir, "roc_no_data.png")

        self.plotter.plot_roc_curves(results, None, output_path)

        # Check that file was created even with no data
        self.assertTrue(os.path.exists(output_path))

    def test_create_metrics_table(self):
        """Test creating a metrics table."""
        results = {
            "accuracy": 0.85,
            "classification_report": {
                "0": {"precision": 0.9, "recall": 0.8, "f1-score": 0.85, "support": 100},
                "1": {"precision": 0.8, "recall": 0.9, "f1-score": 0.85, "support": 100},
                "macro avg": {"precision": 0.85, "recall": 0.85, "f1-score": 0.85, "support": 200},
            },
            "confusion_matrix": np.array([[5, 2], [1, 7]]),
        }
        class_names = ["Class A", "Class B"]
        output_path = os.path.join(self.temp_dir, "metrics_table.png")

        self.plotter.create_metrics_table(results, class_names, output_path)

        # Check that file was created
        self.assertTrue(os.path.exists(output_path))

    def test_plot_ks_curves(self):
        """Test plotting Kolmogorov-Smirnov curves."""
        n_samples = 100
        n_classes = 3
        y_probs = np.random.rand(n_samples, n_classes)
        y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)

        results = {
            "y_probs": y_probs,
            "ks_test": 0.25,
            "ks_p_value": 0.01,
        }
        class_names = ["Class A", "Class B", "Class C"]
        output_path = os.path.join(self.temp_dir, "ks_curves.png")

        self.plotter.plot_ks_curves(results, class_names, output_path)

        num_png_files = len([f for f in os.listdir(self.temp_dir) if f.endswith(".png")])
        self.assertGreater(num_png_files, 0)


if __name__ == "__main__":
    unittest.main()
