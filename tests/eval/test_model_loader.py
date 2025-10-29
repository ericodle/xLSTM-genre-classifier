#!/usr/bin/env python3
"""
Unit tests for model_loader.py.
Tests unified model loading functionality.
"""

import logging
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from src.eval.model_loader import JoblibModelWrapper, ONNXModelWrapper, UnifiedModelLoader

# Create a test logger
test_logger = logging.getLogger("test_logger")


class TestONNXModelWrapper(unittest.TestCase):
    """Test cases for ONNXModelWrapper."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = MagicMock()
        self.mock_session.get_inputs.return_value = [MagicMock(name="input")]
        self.mock_session.get_outputs.return_value = [MagicMock(name="output")]
        self.model_type = "FC"

    def test_onnx_wrapper_init(self):
        """Test ONNX wrapper initialization."""
        wrapper = ONNXModelWrapper(self.mock_session, self.model_type, test_logger)

        self.assertEqual(wrapper.model_type, self.model_type)
        # Input/output names come from session.get_inputs/outputs, which are mocked
        self.assertIsNotNone(wrapper.input_name)
        self.assertIsNotNone(wrapper.output_name)

    def test_onnx_wrapper_eval(self):
        """Test ONNX wrapper eval method."""
        wrapper = ONNXModelWrapper(self.mock_session, self.model_type, test_logger)

        # Should not raise an error
        result = wrapper.eval()
        self.assertIsNone(result)

    def test_onnx_wrapper_call(self):
        """Test ONNX wrapper call method."""
        # Mock session run to return output
        expected_output = np.array([[0.3, 0.7]])
        self.mock_session.run.return_value = [expected_output]

        wrapper = ONNXModelWrapper(self.mock_session, self.model_type, test_logger)

        # Create input tensor
        input_tensor = torch.randn(1, 10)
        result = wrapper(input_tensor)

        # Should convert numpy to torch tensor
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (1, 2))


class TestJoblibModelWrapper(unittest.TestCase):
    """Test cases for JoblibModelWrapper."""

    def test_joblib_wrapper_with_predict_proba(self):
        """Test joblib wrapper with model that has predict_proba."""
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
        model_type = "RandomForest"

        wrapper = JoblibModelWrapper(mock_model, model_type, test_logger)

        # Test call
        input_tensor = torch.randn(1, 10)
        result = wrapper(input_tensor)

        # Should return torch tensor with probabilities
        self.assertIsInstance(result, torch.Tensor)
        mock_model.predict_proba.assert_called_once()

    def test_joblib_wrapper_eval(self):
        """Test joblib wrapper eval method."""
        mock_model = MagicMock()
        wrapper = JoblibModelWrapper(mock_model, "RF", test_logger)

        # Should not raise an error
        result = wrapper.eval()
        self.assertIsNone(result)


class TestUnifiedModelLoader(unittest.TestCase):
    """Test cases for UnifiedModelLoader."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.loader = UnifiedModelLoader()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_joblib_model(self):
        """Test loading a joblib model (skipped - requires actual joblib file)."""
        # Skip this test as it requires actual joblib library loaded
        pass

    def test_load_unsupported_format(self):
        """Test loading a model with unsupported format."""
        # Create a file with unsupported extension
        txt_path = os.path.join(self.temp_dir, "model.txt")

        with self.assertRaises(ValueError) as context:
            self.loader.load_model(txt_path)

        self.assertIn("Unsupported model format", str(context.exception))

    def test_load_nonexistent_file(self):
        """Test loading a nonexistent model file."""
        nonexistent_path = os.path.join(self.temp_dir, "nonexistent.joblib")

        with self.assertRaises(FileNotFoundError):
            self.loader.load_model(nonexistent_path)


if __name__ == "__main__":
    unittest.main()
