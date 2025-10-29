"""
Unified model loading utilities for evaluation.
Supports both neural networks (ONNX) and conventional ML models (joblib).
"""

import logging
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from core.utils import get_device, setup_logging


class UnifiedModelLoader:
    """Unified loader for both ONNX and joblib models."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the unified model loader."""
        self.logger = logger or setup_logging()

    def load_model(self, model_path: str) -> "UnifiedModelWrapper":
        """
        Load a model from either ONNX or joblib format.

        Args:
            model_path: Path to the model file

        Returns:
            UnifiedModelWrapper instance
        """
        if model_path.endswith(".onnx"):
            return self._load_onnx_model(model_path)
        elif model_path.endswith(".joblib"):
            return self._load_joblib_model(model_path)
        else:
            raise ValueError(
                f"Unsupported model format. Expected .onnx or .joblib, got: {model_path}"
            )

    def _load_onnx_model(self, model_path: str) -> "ONNXModelWrapper":
        """Load an ONNX model."""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for ONNX model evaluation. Install it with: pip install onnxruntime"
            )

        # Create inference session
        session = ort.InferenceSession(model_path)

        # Get input and output names
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        # Determine model type from input shape
        input_shape = session.get_inputs()[0].shape
        if len(input_shape) == 2:  # (batch_size, features)
            model_type = "FC"
        elif len(input_shape) == 3:  # (batch_size, time_steps, features)
            model_type = "RNN"
        elif len(input_shape) == 4:  # (batch_size, channels, height, width)
            model_type = "CNN"
        else:
            model_type = "Unknown"

        self.logger.info(f"Loaded ONNX model: {model_path}")
        self.logger.info(f"Model type: {model_type}")
        self.logger.info(f"Input shape: {input_shape}")
        self.logger.info(f"Input name: {input_name}")
        self.logger.info(f"Output name: {output_name}")

        return ONNXModelWrapper(session, model_type, self.logger)

    def _load_joblib_model(self, model_path: str) -> "JoblibModelWrapper":
        """Load a joblib model."""
        try:
            import joblib
        except ImportError:
            raise ImportError(
                "joblib is required for conventional ML model evaluation. Install it with: pip install joblib"
            )

        # Load the model
        model = joblib.load(model_path)

        # Determine model type from the loaded object
        model_type = self._detect_joblib_model_type(model)

        self.logger.info(f"Loaded joblib model: {model_path}")
        self.logger.info(f"Model type: {model_type}")
        self.logger.info(f"Model class: {type(model).__name__}")

        return JoblibModelWrapper(model, model_type, self.logger)

    def _detect_joblib_model_type(self, model: Any) -> str:
        """Detect the type of joblib model."""
        model_class = type(model).__name__

        if "SVM" in model_class or "SVC" in model_class:
            return "SVM"
        elif "RandomForest" in model_class:
            return "RandomForest"
        elif "GaussianNB" in model_class or "NaiveBayes" in model_class:
            return "NaiveBayes"
        elif "KNeighbors" in model_class or "KNN" in model_class:
            return "KNN"
        else:
            return "Unknown"


class UnifiedModelWrapper:
    """Base wrapper for unified model evaluation."""

    def __init__(self, model: Any, model_type: str, logger: logging.Logger):
        """Initialize the model wrapper."""
        self.model = model
        self.model_type = model_type
        self.logger = logger

    def eval(self):
        """Set model to evaluation mode (for neural networks)."""
        if hasattr(self.model, "eval"):
            self.model.eval()

    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Forward pass through the model."""
        raise NotImplementedError("Subclasses must implement __call__ method")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_type": self.model_type,
            "model_class": type(self.model).__name__,
        }


class ONNXModelWrapper(UnifiedModelWrapper):
    """Wrapper for ONNX models."""

    def __init__(self, session: Any, model_type: str, logger: logging.Logger):
        """Initialize ONNX model wrapper."""
        super().__init__(session, model_type, logger)
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name

    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Forward pass through the ONNX model."""
        # Convert to numpy if needed
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()

        # Run inference
        output = self.session.run([self.output_name], {self.input_name: x})[0]
        return torch.from_numpy(output)

    def get_model_info(self) -> Dict[str, Any]:
        """Get ONNX model information."""
        info = super().get_model_info()
        info.update(
            {
                "input_name": self.input_name,
                "output_name": self.output_name,
                "input_shape": self.session.get_inputs()[0].shape,
            }
        )
        return info


class JoblibModelWrapper(UnifiedModelWrapper):
    """Wrapper for joblib models."""

    def __init__(self, model: Any, model_type: str, logger: logging.Logger):
        """Initialize joblib model wrapper."""
        super().__init__(model, model_type, logger)

    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Forward pass through the joblib model."""
        # Convert to numpy if needed
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()

        # Handle different input shapes
        if len(x.shape) == 3:  # (batch_size, time_steps, features)
            # Flatten for conventional ML models
            x = x.reshape(x.shape[0], -1)
        elif len(x.shape) == 4:  # (batch_size, channels, height, width)
            # Flatten for conventional ML models
            x = x.reshape(x.shape[0], -1)

        # Get predictions
        if hasattr(self.model, "predict_proba"):
            # Use predict_proba if available (most conventional ML models)
            predictions = self.model.predict_proba(x)
        elif hasattr(self.model, "decision_function"):
            # Use decision_function for SVM
            predictions = self.model.decision_function(x)
            # Convert to probabilities if needed
            if len(predictions.shape) == 1:  # Binary classification
                predictions = np.column_stack([-predictions, predictions])
            else:  # Multi-class
                predictions = self._decision_function_to_proba(predictions)
        else:
            # Fallback to predict and convert to one-hot
            pred_labels = self.model.predict(x)
            predictions = self._labels_to_proba(pred_labels)

        return torch.from_numpy(predictions.astype(np.float32))

    def _decision_function_to_proba(self, decision_scores: np.ndarray) -> np.ndarray:
        """Convert decision function scores to probabilities."""
        # Simple sigmoid transformation
        exp_scores = np.exp(decision_scores)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def _labels_to_proba(self, labels: np.ndarray) -> np.ndarray:
        """Convert predicted labels to probability matrix."""
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        n_samples = len(labels)

        proba = np.zeros((n_samples, n_classes))
        for i, label in enumerate(labels):
            label_idx = np.where(unique_labels == label)[0][0]
            proba[i, label_idx] = 1.0

        return proba

    def get_model_info(self) -> Dict[str, Any]:
        """Get joblib model information."""
        info = super().get_model_info()

        # Add model-specific information
        if hasattr(self.model, "n_features_in_"):
            info["n_features_in"] = self.model.n_features_in_
        if hasattr(self.model, "classes_"):
            info["classes"] = self.model.classes_.tolist()
        if hasattr(self.model, "n_classes_"):
            info["n_classes"] = self.model.n_classes_

        return info
