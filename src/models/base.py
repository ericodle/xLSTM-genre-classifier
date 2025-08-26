"""
Base model class for all neural network models in GenreDiscern.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import json
import os


class BaseModel(nn.Module, ABC):
    """Abstract base class for all neural network models."""
    
    def __init__(self, model_name: str = "base_model"):
        super().__init__()
        self.model_name = model_name
        self.is_trained = False
        self.training_history = {}
        self.model_config = {}
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            'model_name': self.model_name,
            'model_type': self.__class__.__name__,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'is_trained': self.is_trained,
            'model_config': self.model_config
        }
    
    def save_model(self, filepath: str, save_optimizer: bool = False, optimizer_state: Optional[Dict] = None) -> None:
        """Save the model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        save_dict = {
            'model_state_dict': self.state_dict(),
            'model_config': self.model_config,
            'training_history': self.training_history,
            'is_trained': self.is_trained
        }
        
        if save_optimizer and optimizer_state:
            save_dict['optimizer_state'] = optimizer_state
        
        torch.save(save_dict, filepath)
    
    def load_model(self, filepath: str, load_optimizer: bool = False) -> Optional[Dict]:
        """Load the model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location='cpu')
        
        self.load_state_dict(checkpoint['model_state_dict'])
        self.model_config = checkpoint.get('model_config', {})
        self.training_history = checkpoint.get('training_history', {})
        self.is_trained = checkpoint.get('is_trained', False)
        
        if load_optimizer and 'optimizer_state' in checkpoint:
            return checkpoint['optimizer_state']
        
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
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': total_params - trainable_params
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
    
    def export_to_onnx(self, filepath: str, input_shape: Tuple[int, ...], dynamic_batch: bool = True) -> None:
        """Export the model to ONNX format."""
        try:
            dummy_input = torch.randn(1, *input_shape)
            
            if dynamic_batch:
                # Allow dynamic batch size
                torch.onnx.export(
                    self,
                    dummy_input,
                    filepath,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )
            else:
                torch.onnx.export(
                    self,
                    dummy_input,
                    filepath,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output']
                )
                
        except Exception as e:
            raise RuntimeError(f"Failed to export model to ONNX: {e}")
    
    def set_training_mode(self, mode: bool) -> None:
        """Set the training mode and update the is_trained flag."""
        self.train(mode)
        if not mode:  # If setting to eval mode
            self.is_trained = True 