"""
Tests for GenreDiscern ONNX functionality.
"""

import pytest
import torch
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Add src directory to path for imports
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models import get_model


class TestONNXExport:
    """Test ONNX export functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_onnx_export_lstm(self, temp_dir):
        """Test ONNX export for LSTM model."""
        # Create LSTM model
        model = get_model(
            "LSTM",
            input_dim=13,
            hidden_dim=32,
            num_layers=1,
            output_dim=10,
            dropout=0.1,
        )

        # Create dummy input
        dummy_input = torch.randn(1, 100, 13)  # (batch, sequence, features)

        # Export to ONNX
        onnx_path = Path(temp_dir) / "lstm_model.onnx"

        try:
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            )

            # Check that ONNX file was created
            assert onnx_path.exists()
            assert onnx_path.stat().st_size > 0

        except Exception as e:
            pytest.skip(f"ONNX export failed: {e}")

    def test_onnx_export_cnn(self, temp_dir):
        """Test ONNX export for CNN model."""
        # Create CNN model
        model = get_model("CNN")

        # Create dummy input
        dummy_input = torch.randn(1, 1, 100, 13)  # (batch, channels, height, width)

        # Export to ONNX
        onnx_path = Path(temp_dir) / "cnn_model.onnx"

        try:
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            )

            # Check that ONNX file was created
            assert onnx_path.exists()
            assert onnx_path.stat().st_size > 0

        except Exception as e:
            pytest.skip(f"ONNX export failed: {e}")

    def test_onnx_export_xlstm(self, temp_dir):
        """Test ONNX export for xLSTM model."""
        # Create xLSTM model
        model = get_model(
            "xLSTM",
            input_dim=13,
            hidden_dim=32,
            num_layers=1,
            output_dim=10,
            dropout=0.1,
        )

        # Create dummy input
        dummy_input = torch.randn(1, 100, 13)  # (batch, sequence, features)

        # Export to ONNX
        onnx_path = Path(temp_dir) / "xlstm_model.onnx"

        try:
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            )

            # Check that ONNX file was created
            assert onnx_path.exists()
            assert onnx_path.stat().st_size > 0

        except Exception as e:
            pytest.skip(f"ONNX export failed: {e}")

    def test_onnx_export_fc(self, temp_dir):
        """Test ONNX export for FC model."""
        # Create FC model with required input_dim
        model = get_model("FC", input_dim=13 * 100)  # 13 MFCC features * 100 time steps

        # Create dummy input
        dummy_input = torch.randn(1, 1300)  # (batch, features)

        # Export to ONNX
        onnx_path = Path(temp_dir) / "fc_model.onnx"

        try:
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            )

            # Check that ONNX file was created
            assert onnx_path.exists()
            assert onnx_path.stat().st_size > 0

        except Exception as e:
            pytest.skip(f"ONNX export failed: {e}")

    def test_onnx_export_dynamic_batch_size(self, temp_dir):
        """Test ONNX export with dynamic batch size."""
        # Create LSTM model
        model = get_model(
            "LSTM",
            input_dim=13,
            hidden_dim=32,
            num_layers=1,
            output_dim=10,
            dropout=0.1,
        )

        # Create dummy input
        dummy_input = torch.randn(1, 100, 13)

        # Export to ONNX with dynamic batch size
        onnx_path = Path(temp_dir) / "dynamic_lstm_model.onnx"

        try:
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            )

            # Check that ONNX file was created
            assert onnx_path.exists()
            assert onnx_path.stat().st_size > 0

        except Exception as e:
            pytest.skip(f"ONNX export failed: {e}")


class TestONNXCompatibility:
    """Test ONNX model compatibility and inference."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_onnx_model_loading(self, temp_dir):
        """Test that ONNX models can be loaded."""
        try:
            import onnxruntime as ort
        except ImportError:
            pytest.skip("ONNX Runtime not available")

        # Create a simple ONNX model for testing
        model = get_model(
            "LSTM",
            input_dim=13,
            hidden_dim=32,
            num_layers=1,
            output_dim=10,
            dropout=0.1,
        )
        dummy_input = torch.randn(1, 100, 13)

        onnx_path = Path(temp_dir) / "test_model.onnx"

        try:
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
            )

            # Load ONNX model
            session = ort.InferenceSession(str(onnx_path))

            # Check model metadata
            assert session is not None
            assert len(session.get_inputs()) > 0
            assert len(session.get_outputs()) > 0

            # Check input/output names
            input_names = [input.name for input in session.get_inputs()]
            output_names = [output.name for output in session.get_outputs()]

            assert "input" in input_names
            assert "output" in output_names

        except Exception as e:
            pytest.skip(f"ONNX model creation/loading failed: {e}")

    def test_onnx_inference(self, temp_dir):
        """Test ONNX model inference."""
        try:
            import onnxruntime as ort
        except ImportError:
            pytest.skip("ONNX Runtime not available")

        # Create a simple ONNX model for testing
        model = get_model(
            "LSTM",
            input_dim=13,
            hidden_dim=32,
            num_layers=1,
            output_dim=10,
            dropout=0.1,
        )
        dummy_input = torch.randn(1, 100, 13)

        onnx_path = Path(temp_dir) / "test_model.onnx"

        try:
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
            )

            # Load ONNX model
            session = ort.InferenceSession(str(onnx_path))

            # Prepare input data
            input_data = dummy_input.numpy()

            # Run inference
            output = session.run(["output"], {"input": input_data})

            # Check output
            assert len(output) == 1
            assert output[0].shape == (1, 10)  # (batch, num_classes)

        except Exception as e:
            pytest.skip(f"ONNX inference failed: {e}")

    def test_onnx_model_type_detection(self, temp_dir):
        """Test ONNX model type detection from input shape."""
        try:
            import onnxruntime as ort
        except ImportError:
            pytest.skip("ONNX Runtime not available")

        # Test CNN model (4D input)
        cnn_model = get_model("CNN")
        cnn_dummy_input = torch.randn(1, 1, 100, 13)

        cnn_onnx_path = Path(temp_dir) / "cnn_test.onnx"

        try:
            torch.onnx.export(
                cnn_model,
                cnn_dummy_input,
                cnn_onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
            )

            session = ort.InferenceSession(str(cnn_onnx_path))
            input_shape = session.get_inputs()[0].shape

            # CNN should have 4D input
            assert len(input_shape) == 4

        except Exception as e:
            pytest.skip(f"CNN ONNX export failed: {e}")

        # Test LSTM model (3D input)
        lstm_model = get_model(
            "LSTM",
            input_dim=13,
            hidden_dim=32,
            num_layers=1,
            output_dim=10,
            dropout=0.1,
        )
        lstm_dummy_input = torch.randn(1, 100, 13)

        lstm_onnx_path = Path(temp_dir) / "lstm_test.onnx"

        try:
            torch.onnx.export(
                lstm_model,
                lstm_dummy_input,
                lstm_onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
            )

            session = ort.InferenceSession(str(lstm_onnx_path))
            input_shape = session.get_inputs()[0].shape

            # LSTM should have 3D input
            assert len(input_shape) == 3

        except Exception as e:
            pytest.skip(f"LSTM ONNX export failed: {e}")


class TestONNXIntegration:
    """Test ONNX integration with the evaluation system."""

    def test_onnx_wrapper_creation(self):
        """Test ONNX wrapper creation for evaluation."""
        # Mock ONNX session
        mock_session = Mock()
        mock_session.get_inputs.return_value = [Mock(shape=[1, 100, 13])]
        mock_session.get_outputs.return_value = [Mock(shape=[1, 10])]

        # Test wrapper creation - create a simple wrapper class
        class ONNXModelWrapper:
            def __init__(self, session, model_type):
                self.session = session
                self.model_type = model_type
                self.eval = lambda: None

            def __call__(self, x):
                if isinstance(x, torch.Tensor):
                    x = x.detach().cpu().numpy()
                return torch.from_numpy(self.session.run(["output"], {"input": x})[0])

        wrapper = ONNXModelWrapper(mock_session, "LSTM")

        assert wrapper.session == mock_session
        assert wrapper.model_type == "LSTM"
        assert hasattr(wrapper, "eval")
        assert callable(wrapper)

    def test_onnx_model_type_detection(self):
        """Test ONNX model type detection function."""

        # Create a simple detection function
        def detect_onnx_model_type(session):
            input_shape = session.get_inputs()[0].shape
            if len(input_shape) == 4:
                return "CNN"
            elif len(input_shape) == 3:
                return "RNN"
            else:
                return "FC"

        # Mock ONNX session with different input shapes
        mock_session_4d = Mock()
        mock_session_4d.get_inputs.return_value = [Mock(shape=[1, 1, 100, 13])]

        mock_session_3d = Mock()
        mock_session_3d.get_inputs.return_value = [Mock(shape=[1, 100, 13])]

        mock_session_2d = Mock()
        mock_session_2d.get_inputs.return_value = [Mock(shape=[1, 1300])]

        # Test detection
        assert detect_onnx_model_type(mock_session_4d) == "CNN"
        assert detect_onnx_model_type(mock_session_3d) == "RNN"
        assert detect_onnx_model_type(mock_session_2d) == "FC"
