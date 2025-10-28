"""
{Convolutional Neural Network} model for GenreDiscern.
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
    DEFAULT_CNN_BASE_FILTERS,
    DEFAULT_CNN_CONV_LAYERS,
    DEFAULT_CNN_FC_HIDDEN,
    DEFAULT_CNN_KERNEL_SIZE,
    DEFAULT_CNN_POOL_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_NUM_CLASSES,
)


class CNN_model(BaseModel):
    """2D Convolutional Neural Network model with configurable architecture."""

    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = DEFAULT_NUM_CLASSES,
        dropout: float = DEFAULT_DROPOUT,
        conv_layers: int = DEFAULT_CNN_CONV_LAYERS,
        base_filters: int = DEFAULT_CNN_BASE_FILTERS,
        kernel_size: int = DEFAULT_CNN_KERNEL_SIZE,
        pool_size: int = DEFAULT_CNN_POOL_SIZE,
        fc_hidden: int = DEFAULT_CNN_FC_HIDDEN,
        regression_mode: bool = False,  # NEW: If True, outputs membership scores (not logits)
    ):
        super().__init__(model_name="CNN_model")

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.dropout = dropout
        self.conv_layers = conv_layers
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.fc_hidden = fc_hidden
        self.regression_mode = regression_mode

        # Store configuration
        self.model_config = {
            "input_channels": input_channels,
            "num_classes": num_classes,
            "dropout": dropout,
            "conv_layers": conv_layers,
            "base_filters": base_filters,
            "kernel_size": kernel_size,
            "pool_size": pool_size,
            "fc_hidden": fc_hidden,
            "regression_mode": regression_mode,
        }

        # Check for potentially problematic configurations
        self._check_model_size_warnings()

        # Build configurable CNN architecture
        self.conv_layers_seq = self._build_conv_layers()

        # We'll calculate the flattened size dynamically in forward pass
        self.flatten_size = None
        self.fc_layers = None

    def _build_conv_layers(self):
        """Build configurable convolutional layers."""
        layers = []
        in_channels = self.input_channels

        for i in range(self.conv_layers):
            # Calculate number of filters for this layer (exponential growth)
            out_channels = self.base_filters * (2**i)

            # Convolutional layer
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(self.kernel_size, self.kernel_size),
                    padding=self.kernel_size // 2,
                )
            )
            layers.append(nn.ReLU())

            # Downsample less aggressively: pool every second block (except last)
            # This preserves spatial resolution longer compared to global pooling
            if i % 2 == 1 and i < self.conv_layers - 1:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            # Batch normalization
            layers.append(nn.BatchNorm2d(out_channels))

            # Dropout
            layers.append(nn.Dropout(p=self.dropout))

            in_channels = out_channels

        # Flatten for fully connected layers
        layers.append(nn.Flatten())

        return nn.Sequential(*layers)

    def _check_model_size_warnings(self):
        """Check for potentially problematic model configurations and issue warnings."""
        import warnings

        # Calculate estimated parameters for conv layers
        estimated_conv_params = 0
        in_channels = self.input_channels

        for i in range(self.conv_layers):
            out_channels = self.base_filters * (2**i)
            # Conv2d parameters: (in_channels * out_channels * kernel_size^2) + out_channels (bias)
            conv_params = (
                in_channels * out_channels * self.kernel_size * self.kernel_size
            ) + out_channels
            estimated_conv_params += conv_params
            in_channels = out_channels

        # Estimate FC layer parameters (rough approximation)
        # Assuming input size of 1292x13 (typical for MFCC data)
        estimated_fc_params = (
            self.fc_hidden * 1000 + self.num_classes * self.fc_hidden
        )  # Rough estimate

        total_estimated_params = estimated_conv_params + estimated_fc_params

        # Issue warnings based on model size
        if total_estimated_params > 50_000_000:  # 50M parameters
            warnings.warn(
                f"⚠️  WARNING: Model may be too large! "
                f"Estimated parameters: ~{total_estimated_params:,} "
                f"(conv_layers={self.conv_layers}, base_filters={self.base_filters}). "
                f"This may cause memory issues and slow training. "
                f"Consider reducing conv_layers or base_filters.",
                UserWarning,
            )
        elif total_estimated_params > 10_000_000:  # 10M parameters
            warnings.warn(
                f"⚠️  CAUTION: Large model detected! "
                f"Estimated parameters: ~{total_estimated_params:,} "
                f"(conv_layers={self.conv_layers}, base_filters={self.base_filters}). "
                f"Training may be slow. Monitor GPU memory usage.",
                UserWarning,
            )
        elif total_estimated_params > 1_000_000:  # 1M parameters
            print(
                f"ℹ️  INFO: Model size: ~{total_estimated_params:,} parameters "
                f"(conv_layers={self.conv_layers}, base_filters={self.base_filters})"
            )

        # Check for specific problematic combinations
        if self.conv_layers >= 8:
            warnings.warn(
                f"⚠️  WARNING: {self.conv_layers} conv layers is very deep! "
                f"This may cause vanishing gradients and slow training. "
                f"Consider using 3-5 layers instead.",
                UserWarning,
            )

        if self.base_filters >= 128 and self.conv_layers >= 6:
            warnings.warn(
                f"⚠️  WARNING: High base_filters ({self.base_filters}) with many layers "
                f"({self.conv_layers}) may create an extremely large model! "
                f"Consider reducing base_filters to 32-64.",
                UserWarning,
            )

    def _build_fc_layers(self, flatten_size):
        """Build fully connected layers once we know the flattened size."""
        if self.fc_layers is None or self.flatten_size != flatten_size:
            self.flatten_size = flatten_size
            if self.regression_mode:
                # For regression: output membership scores directly with sigmoid
                self.fc_layers = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(flatten_size, self.fc_hidden),
                    nn.ReLU(),
                    nn.Dropout(p=self.dropout),
                    nn.Linear(self.fc_hidden, self.num_classes),
                    nn.Sigmoid(),  # Membership scores in [0, 1]
                )
            else:
                # For classification: output logits for softmax
                self.fc_layers = nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(flatten_size, self.fc_hidden),
                    nn.ReLU(),
                    nn.Dropout(p=self.dropout),
                    nn.Linear(self.fc_hidden, self.num_classes),  # Raw logits for CrossEntropyLoss
                )
            # Move FC layers to the same device as the model
            if hasattr(self, "conv_layers_seq"):
                device = next(self.conv_layers_seq.parameters()).device
                self.fc_layers = self.fc_layers.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the 2D CNN."""
        # Input shape: (batch, mfcc_features) or (batch, time_steps, mfcc_features)
        # Need to reshape to (batch, channels, height, width) for 2D conv

        if len(x.shape) == 2:
            # Batch: (batch, mfcc_features) -> (batch, 1, 1, mfcc_features)
            x = x.unsqueeze(1).unsqueeze(1)
        elif len(x.shape) == 3:
            # Batch: (batch, time_steps, mfcc_features) -> (batch, 1, time_steps, mfcc_features)
            x = x.unsqueeze(1)
        else:
            # Already in correct format: (batch, 1, height, width)
            pass

        # Apply 2D convolutions
        x = self.conv_layers_seq(x)

        # Calculate flattened size and build FC layers if needed
        # x shape after conv_layers: (batch, channels, height, width)
        if len(x.shape) == 4:
            flatten_size = x.shape[1] * x.shape[2] * x.shape[3]
        else:
            # If somehow we don't have 4D, use the total size
            flatten_size = x.numel() // x.shape[0] if x.shape[0] > 0 else x.numel()

        self._build_fc_layers(flatten_size)

        # Apply fully connected layers
        if self.fc_layers is None:
            raise RuntimeError("FC layers not initialized. Call _build_fc_layers first.")
        x = self.fc_layers(x)
        return torch.as_tensor(x)
