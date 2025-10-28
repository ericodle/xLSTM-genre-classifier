"""
Base model class for all neural network models in GenreDiscern.
"""

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """Abstract base class for all neural network models."""

    def __init__(self, model_name: str = "base_model"):
        super().__init__()
        self.model_name = model_name
        self.is_trained = False
        self.training_history: dict[str, Any] = {}
        self.model_config: dict[str, Any] = {}

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "model_name": self.model_name,
            "model_type": self.__class__.__name__,
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "is_trained": self.is_trained,
            "model_config": self.model_config,
        }

    def save_model(
        self,
        filepath: str,
        input_shape: Tuple[int, ...],
        save_optimizer: bool = False,
        optimizer_state: Optional[Dict] = None,
    ) -> None:
        """Save the model to ONNX format."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Ensure filepath has .onnx extension
        if not filepath.endswith(".onnx"):
            filepath = filepath.replace(".pth", ".onnx")

        # Export to ONNX
        self.export_to_onnx(filepath, input_shape, dynamic_batch=True)

        # Save metadata separately as JSON
        metadata_path = filepath.replace(".onnx", "_metadata.json")
        # Convert training history to serializable format
        serializable_training_history = {}
        for key, value in self.training_history.items():
            if isinstance(value, list):
                # Convert any tensors in the list to regular numbers
                serializable_training_history[key] = [
                    item.item() if hasattr(item, "item") else item for item in value
                ]
            else:
                serializable_training_history[key] = value

        # Convert model config to serializable format
        serializable_model_config = {}
        for key, value in self.model_config.items():
            if hasattr(value, "item"):  # Tensor
                serializable_model_config[key] = value.item()
            elif isinstance(value, (list, tuple)):
                serializable_model_config[key] = [
                    item.item() if hasattr(item, "item") else item for item in value
                ]
            else:
                serializable_model_config[key] = value

        metadata = {
            "model_config": serializable_model_config,
            "training_history": serializable_training_history,
            "is_trained": self.is_trained,
            "input_shape": list(input_shape),  # Convert tuple to list for JSON serialization
            "model_type": self.__class__.__name__,
        }

        if save_optimizer and optimizer_state:
            # Convert optimizer state to serializable format
            def convert_tensor_to_serializable(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.cpu().tolist()
                elif isinstance(obj, dict):
                    return {k: convert_tensor_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_tensor_to_serializable(item) for item in obj]
                else:
                    return obj

            serializable_optimizer_state = convert_tensor_to_serializable(optimizer_state)
            metadata["optimizer_state"] = serializable_optimizer_state

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def load_model(self, filepath: str, load_optimizer: bool = False) -> Optional[Dict]:
        """Load model metadata from ONNX model."""
        if not filepath.endswith(".onnx"):
            filepath = filepath.replace(".pth", ".onnx")

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"ONNX model file not found: {filepath}")

        # Load metadata
        metadata_path = filepath.replace(".onnx", "_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            self.model_config = metadata.get("model_config", {})
            self.training_history = metadata.get("training_history", {})
            self.is_trained = metadata.get("is_trained", False)

            if load_optimizer and "optimizer_state" in metadata:
                return dict(metadata["optimizer_state"])
        else:
            # If no metadata file, set defaults
            self.model_config = {}
            self.training_history = {}
            self.is_trained = True  # Assume trained if ONNX exists

        return None

    def count_parameters(self) -> Dict[str, int]:
        """Count the number of parameters in the model."""
        total_params = 0
        trainable_params = 0

        for param in self.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        return {
            "total": total_params,
            "trainable": trainable_params,
            "non_trainable": total_params - trainable_params,
        }

    def freeze_layers(self, layer_names: list) -> None:
        """Freeze specific layers of the model."""
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False

    def unfreeze_layers(self, layer_names: list) -> None:
        """Unfreeze specific layers of the model."""
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True

    def get_layer_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Get the output shape of the model for a given input shape."""
        # Create a dummy input tensor
        dummy_input = torch.randn(1, *input_shape)

        # Register a hook to capture the output shape
        output_shape = None

        def hook_fn(module, input, output):
            nonlocal output_shape
            output_shape = output.shape[1:]  # Remove batch dimension

        # Register the hook on the last layer
        last_layer = list(self.children())[-1]
        hook = last_layer.register_forward_hook(hook_fn)

        # Forward pass
        with torch.no_grad():
            self(dummy_input)

        # Remove the hook
        hook.remove()

        return output_shape if output_shape else input_shape

    def summary(self) -> str:
        """Get a summary of the model architecture."""
        param_counts = self.count_parameters()

        summary = f"""
Model Summary: {self.model_name}
{'=' * 50}
Total Parameters: {param_counts['total']:,}
Trainable Parameters: {param_counts['trainable']:,}
Non-trainable Parameters: {param_counts['non_trainable']:,}
Model Type: {self.__class__.__name__}
Trained: {self.is_trained}
        """

        return summary

    def export_to_onnx(
        self, filepath: str, input_shape: Tuple[int, ...], dynamic_batch: bool = True
    ) -> None:
        """Export the model to ONNX format."""
        try:
            # Set model to eval mode for export
            self.eval()

            # Move model to CPU for ONNX export
            device = next(self.parameters()).device
            model_cpu = self.cpu()
            dummy_input = torch.randn(1, *input_shape)

            if dynamic_batch:
                # Allow dynamic batch size
                torch.onnx.export(
                    model_cpu,
                    dummy_input,
                    filepath,
                    export_params=True,
                    opset_version=14,  # Updated for ViT attention support
                    do_constant_folding=True,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={
                        "input": {0: "batch_size"},
                        "output": {0: "batch_size"},
                    },
                )
            else:
                torch.onnx.export(
                    model_cpu,
                    dummy_input,
                    filepath,
                    export_params=True,
                    opset_version=14,  # Updated for ViT attention support
                    do_constant_folding=True,
                    input_names=["input"],
                    output_names=["output"],
                )

            # Move model back to original device
            self.to(device)

        except Exception as e:
            # Ensure model is back on original device even if export fails
            self.to(device)
            raise RuntimeError(f"Failed to export model to ONNX: {e}")

    @staticmethod
    def create_onnx_inference_session(filepath: str):
        """Create an ONNX inference session for the model."""
        try:
            import onnxruntime as ort

            return ort.InferenceSession(filepath)
        except ImportError:
            raise ImportError("ONNX Runtime not available. Install with: pip install onnxruntime")
        except Exception as e:
            raise RuntimeError(f"Failed to create ONNX inference session: {e}")

    @staticmethod
    def detect_onnx_model_type(session) -> str:
        """Detect model type from ONNX session input shape."""
        try:
            input_shape = session.get_inputs()[0].shape

            if len(input_shape) == 4:  # [batch, channels, height, width]
                return "CNN"
            elif len(input_shape) == 3:  # [batch, sequence, features]
                return "RNN"
            elif len(input_shape) == 2:  # [batch, features]
                return "FC"
            else:
                return "Unknown"
        except Exception:
            return "Unknown"

    def set_training_mode(self, mode: bool) -> None:
        """Set the training mode and update the is_trained flag."""
        self.train(mode)
        if not mode:  # If setting to eval mode
            self.is_trained = True
