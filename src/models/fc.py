"""
{Fully Connected} model for GenreDiscern.
"""

import os

# Add src directory to path for imports
import sys
from typing import List, Optional

import torch
import torch.nn as nn

from .base import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.constants import (
    DEFAULT_FC_DROPOUT,
    DEFAULT_FC_HIDDEN_DIMS,
    DEFAULT_FC_INPUT_DIM,
    DEFAULT_NUM_CLASSES,
)


class FC_model(BaseModel):
    """Fully Connected Neural Network model."""

    def __init__(
        self,
        input_dim: int = DEFAULT_FC_INPUT_DIM,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = DEFAULT_NUM_CLASSES,
        dropout: float = DEFAULT_FC_DROPOUT,
    ):
        super().__init__(model_name="FC_model")

        if hidden_dims is None:
            hidden_dims = DEFAULT_FC_HIDDEN_DIMS
        # Normalize hidden_dims to a list if an int (or other scalar) was provided
        if isinstance(hidden_dims, (int, float)):
            hidden_dims = [int(hidden_dims)]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout

        # Store configuration
        self.model_config = {
            "input_dim": input_dim,
            "hidden_dims": hidden_dims,
            "output_dim": output_dim,
            "dropout": dropout,
        }

        # Check for potentially problematic configurations
        self._check_fc_model_size_warnings()

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(p=dropout)])
            prev_dim = hidden_dim

        # Output layer - raw logits for CrossEntropyLoss
        layers.append(nn.Linear(prev_dim, output_dim))

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the FC network."""
        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        return torch.as_tensor(self.fc_layers(x))

    def _check_fc_model_size_warnings(self):
        """Check for potentially problematic FC model configurations and issue warnings."""
        import warnings

        # Calculate estimated parameters
        estimated_params = 0
        prev_dim = self.input_dim

        for hidden_dim in self.hidden_dims:
            # Linear layer parameters: (input_dim * output_dim) + output_dim (bias)
            estimated_params += (prev_dim * hidden_dim) + hidden_dim
            prev_dim = hidden_dim

        # Output layer
        estimated_params += (prev_dim * self.output_dim) + self.output_dim

        # Issue warnings based on model size
        if estimated_params > 50_000_000:  # 50M parameters
            warnings.warn(
                f"⚠️  WARNING: FC model may be too large! "
                f"Estimated parameters: ~{estimated_params:,} "
                f"(input_dim={self.input_dim}, hidden_dims={self.hidden_dims}). "
                f"This may cause memory issues and slow training. "
                f"Consider reducing hidden_dims or input_dim.",
                UserWarning,
            )
        elif estimated_params > 10_000_000:  # 10M parameters
            warnings.warn(
                f"⚠️  CAUTION: Large FC model detected! "
                f"Estimated parameters: ~{estimated_params:,} "
                f"(input_dim={self.input_dim}, hidden_dims={self.hidden_dims}). "
                f"Training may be slow. Monitor GPU memory usage.",
                UserWarning,
            )
        elif estimated_params > 1_000_000:  # 1M parameters
            print(
                f"ℹ️  INFO: FC model size: ~{estimated_params:,} parameters "
                f"(input_dim={self.input_dim}, hidden_dims={self.hidden_dims})"
            )

        # Check for specific problematic combinations
        if len(self.hidden_dims) > 5:
            warnings.warn(
                f"⚠️  WARNING: {len(self.hidden_dims)} hidden layers is very deep! "
                f"This may cause vanishing gradients and slow training. "
                f"Consider using 2-4 layers instead.",
                UserWarning,
            )

        if any(dim > 1000 for dim in self.hidden_dims) and len(self.hidden_dims) > 3:
            warnings.warn(
                f"⚠️  WARNING: Large hidden dimensions ({self.hidden_dims}) with many layers "
                f"may create an extremely large model! "
                f"Consider reducing hidden dimensions to 128-512.",
                UserWarning,
            )
