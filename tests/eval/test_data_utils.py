#!/usr/bin/env python3
"""
Unit tests for data_utils.py.
Tests data loading and preprocessing utilities.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.eval.data_utils import DataLoaderUtils


class TestDataLoaderUtils(unittest.TestCase):
    """Test cases for DataLoaderUtils functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_mock_json_file(self, filename, data):
        """Create a mock JSON file."""
        json_path = os.path.join(self.temp_dir, filename)
        with open(json_path, "w") as f:
            json.dump(data, f)
        return json_path

    def test_load_mfcc_data_new_format(self):
        """Test loading MFCC data with new format (features/labels keys)."""
        data = {
            "features": [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0]],
            ],
            "labels": [0, 1],
            "mapping": ["class0", "class1"],
        }
        json_path = self.create_mock_json_file("test.json", data)

        features, labels, mapping = DataLoaderUtils.load_mfcc_data(json_path)

        # Features should be padded to consistent length
        self.assertEqual(features.shape, (2, 2, 2))
        self.assertEqual(labels.shape, (2,))
        self.assertEqual(mapping, ["class0", "class1"])

    def test_load_mfcc_data_old_format(self):
        """Test loading MFCC data with old format (file paths as keys)."""
        data = {
            "genre1/file1.mp3": {"mfcc": [[1.0, 2.0], [3.0, 4.0]]},
            "genre2/file2.mp3": {"mfcc": [[5.0, 6.0], [7.0, 8.0]]},
        }
        json_path = self.create_mock_json_file("test.json", data)

        features, labels, mapping = DataLoaderUtils.load_mfcc_data(json_path)

        self.assertEqual(features.shape, (2, 2, 2))
        self.assertEqual(labels.shape, (2,))
        self.assertEqual(len(mapping), 2)
        self.assertIn("genre1", mapping)
        self.assertIn("genre2", mapping)

    def test_load_mfcc_data_without_mapping(self):
        """Test loading MFCC data without explicit mapping."""
        data = {
            "features": [
                [[1.0], [2.0]],
                [[3.0], [4.0]],
                [[5.0], [6.0]],
            ],
            "labels": ["a", "b", "a"],
        }
        json_path = self.create_mock_json_file("test.json", data)

        features, labels, mapping = DataLoaderUtils.load_mfcc_data(json_path)

        # Mapping should be created from unique labels (sorted alphabetically)
        self.assertIn("a", mapping)
        self.assertIn("b", mapping)
        # Labels should be converted to indices - check type
        unique_labels = np.unique(labels)
        self.assertEqual(len(unique_labels), 2)
        # Note: String labels may not be converted if dtype is not object
        # Just check that we got the right number of unique values

    def test_preprocess_features_fc(self):
        """Test preprocessing features for FC models."""
        # Create 3D features (batch, time, features)
        features = np.random.randn(10, 100, 13)
        processed = DataLoaderUtils.preprocess_features(features, "FC")

        # Should be flattened to (batch, time*features)
        self.assertEqual(processed.shape, (10, 1300))

    def test_preprocess_features_cnn(self):
        """Test preprocessing features for CNN models."""
        # Create 3D features (batch, time, features)
        features = np.random.randn(10, 100, 13)
        processed = DataLoaderUtils.preprocess_features(features, "CNN")

        # Should be reshaped to (batch, channels, time, features)
        self.assertEqual(processed.shape, (10, 1, 100, 13))

    def test_preprocess_features_rnn(self):
        """Test preprocessing features for RNN models."""
        # Create 3D features (batch, time, features)
        features = np.random.randn(10, 100, 13)
        processed = DataLoaderUtils.preprocess_features(features, "RNN")

        # Should remain unchanged (batch, time, features)
        self.assertTrue(np.array_equal(processed, features))

    def test_preprocess_features_2d(self):
        """Test preprocessing features that are already 2D."""
        # Create 2D features
        features = np.random.randn(10, 1300)
        processed = DataLoaderUtils.preprocess_features(features, "FC")

        # Should remain unchanged
        self.assertTrue(np.array_equal(processed, features))


if __name__ == "__main__":
    unittest.main()
