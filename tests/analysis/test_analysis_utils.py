#!/usr/bin/env python3
"""
Unit tests for analysis functionality.
Tests the shared utilities and core analysis functions.
"""

import json
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.analysis.utils import (
    AnalysisLogger,
    calculate_statistics,
    ensure_output_directory,
    format_percentage,
    get_dataset_colors,
    get_model_display_name,
    get_model_order,
    infer_dataset_from_path,
    infer_model_from_path,
    load_json_data,
    safe_divide,
    save_json_data,
    setup_logging,
    setup_plotting_style,
)


class TestAnalysisUtils(unittest.TestCase):
    """Test cases for analysis utilities."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data = {"test": "value", "number": 42}

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_setup_logging(self):
        """Test logging setup."""
        logger = setup_logging()
        self.assertIsInstance(logger, logging.Logger)
        # Note: logger.level might be 0 (NOTSET) if no handlers are configured
        # We just check that it's a valid logger instance

    def test_setup_plotting_style(self):
        """Test plotting style setup."""
        # Should not raise any exceptions
        setup_plotting_style()

    def test_infer_dataset_from_path(self):
        """Test dataset inference from path."""
        self.assertEqual(infer_dataset_from_path("gtzan-data/test"), "GTZAN")
        self.assertEqual(infer_dataset_from_path("fma-data/test"), "FMA")
        self.assertEqual(infer_dataset_from_path("unknown-path"), "UNKNOWN")

    def test_infer_model_from_path(self):
        """Test model inference from path."""
        self.assertEqual(infer_model_from_path("fc-model"), "FC")
        self.assertEqual(infer_model_from_path("cnn-model"), "CNN")
        self.assertEqual(infer_model_from_path("lstm-model"), "LSTM")
        self.assertEqual(infer_model_from_path("xlstm-model"), "XLSTM")
        self.assertEqual(infer_model_from_path("tr-model"), "TRANSFORMER")
        self.assertEqual(infer_model_from_path("unknown-model"), "UNKNOWN")

    def test_infer_model_from_path_with_data(self):
        """Test model inference from path with data."""
        data = {"model_type": "CNN"}
        self.assertEqual(infer_model_from_path("unknown", data), "CNN")

        data = {"params": {"kernel": "rbf"}}
        self.assertEqual(infer_model_from_path("unknown", data), "SVM")

    def test_get_model_display_name(self):
        """Test model display name mapping."""
        self.assertEqual(get_model_display_name("TRANSFORMER"), "TR")
        self.assertEqual(get_model_display_name("XLSTM"), "XLSTM")
        self.assertEqual(get_model_display_name("FC"), "FC")

    def test_get_model_order(self):
        """Test model order list."""
        order = get_model_order()
        self.assertIsInstance(order, list)
        self.assertIn("SVM", order)
        self.assertIn("FC", order)
        self.assertIn("CNN", order)

    def test_get_dataset_colors(self):
        """Test dataset color mapping."""
        colors = get_dataset_colors()
        self.assertIsInstance(colors, dict)
        self.assertIn("GTZAN", colors)
        self.assertIn("FMA", colors)

    def test_safe_divide(self):
        """Test safe division function."""
        self.assertEqual(safe_divide(10, 2), 5.0)
        self.assertEqual(safe_divide(10, 0), 0.0)
        self.assertEqual(safe_divide(10, 0, default=1.0), 1.0)

    def test_format_percentage(self):
        """Test percentage formatting."""
        self.assertEqual(format_percentage(0.1234), "12.3%")
        self.assertEqual(format_percentage(0.1234, decimals=2), "12.34%")

    def test_calculate_statistics(self):
        """Test statistics calculation."""
        values = [1, 2, 3, 4, 5]
        stats = calculate_statistics(values)

        self.assertAlmostEqual(stats["mean"], 3.0)
        self.assertAlmostEqual(stats["std"], 1.41, places=1)  # Corrected expected value
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 5.0)
        self.assertEqual(stats["count"], 5)

    def test_calculate_statistics_empty(self):
        """Test statistics calculation with empty list."""
        stats = calculate_statistics([])
        self.assertEqual(stats["mean"], 0.0)
        self.assertEqual(stats["std"], 0.0)
        self.assertEqual(stats["count"], 0)

    def test_ensure_output_directory(self):
        """Test output directory creation."""
        test_path = os.path.join(self.temp_dir, "test", "output")
        result = ensure_output_directory(test_path)

        self.assertTrue(os.path.exists(test_path))
        self.assertEqual(str(result), test_path)

    def test_load_json_data(self):
        """Test JSON data loading."""
        test_file = os.path.join(self.temp_dir, "test.json")
        with open(test_file, "w") as f:
            json.dump(self.test_data, f)

        loaded_data = load_json_data(test_file)
        self.assertEqual(loaded_data, self.test_data)

    def test_load_json_data_file_not_found(self):
        """Test JSON loading with non-existent file."""
        with self.assertRaises(FileNotFoundError):
            load_json_data("non_existent.json")

    def test_load_json_data_invalid_json(self):
        """Test JSON loading with invalid JSON."""
        test_file = os.path.join(self.temp_dir, "invalid.json")
        with open(test_file, "w") as f:
            f.write("invalid json content")

        with self.assertRaises(ValueError):
            load_json_data(test_file)

    def test_save_json_data(self):
        """Test JSON data saving."""
        test_file = os.path.join(self.temp_dir, "test", "output.json")
        save_json_data(self.test_data, test_file)

        self.assertTrue(os.path.exists(test_file))
        with open(test_file, "r") as f:
            loaded_data = json.load(f)
        self.assertEqual(loaded_data, self.test_data)

    def test_analysis_logger(self):
        """Test analysis logger."""
        logger = AnalysisLogger("test")
        self.assertIsInstance(logger.logger, logging.Logger)

        # Test that methods don't raise exceptions
        logger.info("Test info message")
        logger.warning("Test warning message")
        logger.error("Test error message")
        logger.debug("Test debug message")


class TestAnalysisIntegration(unittest.TestCase):
    """Integration tests for analysis functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_analysis_workflow(self):
        """Test complete analysis workflow."""
        # Create mock results data
        results_data = {
            "train": {"accuracy": 0.95},
            "val": {"accuracy": 0.90},
            "test": {"accuracy": 0.88},
        }

        # Create mock directory structure
        model_dir = os.path.join(self.temp_dir, "fc-gtzan-test")
        os.makedirs(model_dir)

        results_file = os.path.join(model_dir, "results.json")
        with open(results_file, "w") as f:
            json.dump(results_data, f)

        # Test that we can load and process the data
        data = load_json_data(results_file)
        self.assertEqual(data["test"]["accuracy"], 0.88)

        # Test model inference
        model = infer_model_from_path(model_dir, data)
        dataset = infer_dataset_from_path(model_dir)

        self.assertEqual(model, "FC")
        self.assertEqual(dataset, "GTZAN")

    @patch("src.analysis.utils.plt")
    def test_plotting_setup(self, mock_plt):
        """Test plotting setup with mocked matplotlib."""
        setup_plotting_style()

        # Verify that rcParams was updated
        mock_plt.rcParams.update.assert_called_once()


if __name__ == "__main__":
    unittest.main()
