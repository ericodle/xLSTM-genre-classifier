"""
Neural network models for GenreDiscern.
"""

import torch
import torch.nn as nn
from typing import List, Optional

from .base import BaseModel

# Add src directory to path for imports
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.constants import (
    DEFAULT_NUM_CLASSES,
    DEFAULT_FC_HIDDEN_DIMS,
    DEFAULT_FC_DROPOUT,
    DEFAULT_CNN_DROPOUT,
    DEFAULT_TRANSFORMER_HEADS,
    DEFAULT_TRANSFORMER_FF_DIM,
)


class FC_model(BaseModel):
    """Fully Connected Neural Network model."""

    def __init__(
        self,
        input_dim: int = 16796,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = DEFAULT_NUM_CLASSES,
        dropout: float = DEFAULT_FC_DROPOUT,
    ):
        super().__init__(model_name="FC_model")

        if hidden_dims is None:
            hidden_dims = DEFAULT_FC_HIDDEN_DIMS

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

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(p=dropout)]
            )
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Softmax(dim=1))

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the FC network."""
        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        return torch.as_tensor(self.fc_layers(x))


class CNN_model(BaseModel):
    """2D Convolutional Neural Network model with dynamic dimension calculation."""

    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = DEFAULT_NUM_CLASSES,
        dropout: float = DEFAULT_CNN_DROPOUT,
    ):
        super().__init__(model_name="CNN_model")

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.dropout = dropout

        # Store configuration
        self.model_config = {
            "input_channels": input_channels,
            "num_classes": num_classes,
            "dropout": dropout,
        }

        # Very simple 2D CNN architecture
        # Input: (batch, channels, height, width)

        self.conv_layers = nn.Sequential(
            # Single conv layer
            nn.Conv2d(self.input_channels, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),  # More conservative pooling
            nn.BatchNorm2d(16),
            nn.Dropout(p=self.dropout),
            # Second conv layer
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),  # More conservative pooling
            nn.BatchNorm2d(32),
            nn.Dropout(p=self.dropout),
            # Third conv layer
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),  # More conservative pooling
            nn.BatchNorm2d(64),
            nn.Dropout(p=self.dropout),
            # Flatten for fully connected layers
            nn.Flatten(),
        )

        # We'll calculate the flattened size dynamically in forward pass
        self.flatten_size = None
        self.fc_layers = None

    def _build_fc_layers(self, flatten_size):
        """Build fully connected layers once we know the flattened size."""
        if self.fc_layers is None or self.flatten_size != flatten_size:
            self.flatten_size = flatten_size
            self.fc_layers = nn.Sequential(
                nn.ReLU(),
                nn.Linear(flatten_size, 64),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
                nn.Linear(64, self.num_classes),
                nn.Softmax(dim=1),
            )
            # Move FC layers to the same device as the model
            if hasattr(self, "conv_layers"):
                device = next(self.conv_layers.parameters()).device
                self.fc_layers = self.fc_layers.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the 2D CNN."""
        # Input shape: (batch, time_steps, mfcc_features)
        # Need to reshape to (batch, channels, height, width) for 2D conv

        if len(x.shape) == 2:
            # Single sample: (time_steps, mfcc_features) -> (1, 1, time_steps, mfcc_features)
            x = x.unsqueeze(0).unsqueeze(0)
        elif len(x.shape) == 3:
            # Batch: (batch, time_steps, mfcc_features) -> (batch, 1, time_steps, mfcc_features)
            x = x.unsqueeze(1)

        # Apply 2D convolutions
        x = self.conv_layers(x)

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
            raise RuntimeError(
                "FC layers not initialized. Call _build_fc_layers first."
            )
        x = self.fc_layers(x)
        return torch.as_tensor(x)


class LSTM_model(BaseModel):
    """Long Short-Term Memory model."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        layer_dim: int,
        output_dim: int,
        dropout_prob: float,
    ):
        super().__init__(model_name="LSTM_model")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob

        # Store configuration
        self.model_config = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "layer_dim": layer_dim,
            "output_dim": output_dim,
            "dropout_prob": dropout_prob,
        }

        self.rnn = nn.LSTM(
            input_dim,
            hidden_dim,
            layer_dim,
            batch_first=True,
            bidirectional=False,
            dropout=dropout_prob,
        )

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the LSTM."""
        # Ensure input is 3D: (batch_size, sequence_length, input_dim)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension

        # Initialize hidden state
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)

        # Forward propagate LSTM
        out, _ = self.rnn(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        out = self.softmax(out)

        return torch.as_tensor(out)


class GRU_model(BaseModel):
    """Gated Recurrent Unit model."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        layer_dim: int,
        output_dim: int,
        dropout_prob: float,
    ):
        super().__init__(model_name="GRU_model")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob

        # Store configuration
        self.model_config = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "layer_dim": layer_dim,
            "output_dim": output_dim,
            "dropout_prob": dropout_prob,
        }

        self.rnn = nn.GRU(
            input_dim,
            hidden_dim,
            layer_dim,
            batch_first=True,
            bidirectional=False,
            dropout=dropout_prob,
        )

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the GRU."""
        # Ensure input is 3D: (batch_size, sequence_length, input_dim)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension

        # Initialize hidden state
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)

        # Forward propagate GRU
        out, _ = self.rnn(x, h0)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        out = self.softmax(out)

        return torch.as_tensor(out)
