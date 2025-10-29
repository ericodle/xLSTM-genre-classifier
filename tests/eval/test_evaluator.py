#!/usr/bin/env python3
"""
Unit tests for evaluator.py.
Tests the ModelEvaluator class.
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from src.eval.evaluator import ModelEvaluator


class TestModelEvaluator(unittest.TestCase):
    """Test cases for ModelEvaluator functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("src.eval.evaluator.UnifiedModelLoader")
    def test_evaluator_init_from_path(self, mock_loader_class):
        """Test ModelEvaluator initialization from model path."""
        # Mock the loader
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_model.model_type = "FC"
        mock_model.eval = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_loader.load_model.return_value = mock_model
        mock_loader_class.return_value = mock_loader

        evaluator = ModelEvaluator("model.onnx")

        # Check that model was loaded
        mock_loader.load_model.assert_called_once_with("model.onnx")
        self.assertIsNotNone(evaluator.model)

    def test_evaluator_init_from_model_object(self):
        """Test ModelEvaluator initialization from model object."""
        mock_model = MagicMock()
        mock_model.eval = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)

        evaluator = ModelEvaluator(mock_model)

        # Check that methods were called on the model
        mock_model.eval.assert_called_once()
        mock_model.to.assert_called_once()

    @patch("src.eval.evaluator.UnifiedModelLoader")
    def test_evaluate_with_basic_data(self, mock_loader_class):
        """Test evaluation with basic mock data."""
        # Setup mock model
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_model.model_type = "FC"
        mock_model.eval = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)

        # Mock model output - batch of predictions
        mock_output = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        mock_model.side_effect = [mock_output]

        mock_loader.load_model.return_value = mock_model
        mock_loader_class.return_value = mock_loader

        # Create evaluator
        evaluator = ModelEvaluator("model.onnx")

        # Create mock dataloader
        mock_dataloader = MagicMock()
        mock_dataloader.__iter__ = MagicMock(
            return_value=iter([(torch.randn(2, 10), torch.tensor([1, 0]))])
        )

        # Run evaluation
        results = evaluator.evaluate_model(mock_dataloader)

        # Check that basic results are present
        self.assertIn("accuracy", results)
        self.assertIn("y_pred", results)  # predictions stored as y_pred
        self.assertIn("y_probs", results)  # probabilities stored as y_probs

    @patch("src.eval.evaluator.UnifiedModelLoader")
    def test_generate_plots(self, mock_loader_class):
        """Test plot generation."""
        # Setup mock model
        mock_loader = MagicMock()
        mock_model = MagicMock()
        mock_model.model_type = "FC"
        mock_model.eval = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_loader.load_model.return_value = mock_model
        mock_loader_class.return_value = mock_loader

        # Create evaluator
        evaluator = ModelEvaluator("model.onnx")

        # Mock evaluation results
        results = {
            "predictions": np.array([0, 1, 1, 0]),
            "probabilities": np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.85, 0.15]]),
            "labels": np.array([0, 1, 1, 0]),
            "confusion_matrix": np.array([[2, 0], [0, 2]]),
            "roc_auc": 1.0,
        }

        output_dir = self.temp_dir
        class_names = ["Class A", "Class B"]

        # Generate plots
        evaluator.generate_evaluation_plots(results, output_dir, class_names)

        # Check that output directory contains plot files
        files = os.listdir(output_dir)
        self.assertTrue(
            any("confusion" in f.lower() for f in files)
            or any("roc" in f.lower() for f in files)
            or len(files) > 0
        )


if __name__ == "__main__":
    unittest.main()
