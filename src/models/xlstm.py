"""
Extended LSTM models for GenreDiscern.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base import BaseModel


class SimpleCausalConv1D(nn.Module):
    """Simplified causal convolution for sequence data."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=self.padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x is (batch, sequence, features)
        # For single timestep, reshape to (batch, features, 1)
        if x.shape[1] == 1:
            # Single timestep: (batch, 1, features) -> (batch, features, 1)
            x = x.transpose(1, 2)
        else:
            # Multiple timesteps: (batch, sequence, features) -> (batch, features, sequence)
            x = x.transpose(1, 2)

        x = self.conv(x)
        # Remove padding to maintain causality
        x = x[:, :, : -self.padding]

        # Transpose back to original format
        if x.shape[2] == 1:
            # Single timestep: (batch, features, 1) -> (batch, 1, features)
            x = x.transpose(1, 2)
        else:
            # Multiple timesteps: (batch, features, sequence) -> (batch, sequence, features)
            x = x.transpose(1, 2)

        return x


class SimpleBlockDiagonal(nn.Module):
    """Simplified block diagonal linear layer."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.block = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(self.block(x))


class SimpleSLSTMBlock(nn.Module):
    """Simplified sLSTM block - core LSTM without causal convolution for single timesteps."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_norm = nn.LayerNorm(input_size)

        # Core LSTM gates
        self.Wz = SimpleBlockDiagonal(input_size, hidden_size)
        self.Wi = SimpleBlockDiagonal(input_size, hidden_size)
        self.Wf = SimpleBlockDiagonal(input_size, hidden_size)
        self.Wo = SimpleBlockDiagonal(input_size, hidden_size)

        self.Rz = SimpleBlockDiagonal(hidden_size, hidden_size)
        self.Ri = SimpleBlockDiagonal(hidden_size, hidden_size)
        self.Rf = SimpleBlockDiagonal(hidden_size, hidden_size)
        self.Ro = SimpleBlockDiagonal(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, state: tuple) -> tuple:
        h_prev, c_prev = state
        x_norm = self.layer_norm(x)

        # Standard LSTM equations without convolution
        z = torch.tanh(self.Wz(x) + self.Rz(h_prev))
        i = torch.sigmoid(self.Wi(x_norm) + self.Ri(h_prev))
        f = torch.sigmoid(self.Wf(x_norm) + self.Rf(h_prev))
        o = torch.sigmoid(self.Wo(x) + self.Ro(h_prev))

        c_t = f * c_prev + i * z
        h_t = o * torch.tanh(c_t)

        return h_t, (h_t, c_t)


class SimpleMLSTMBlock(nn.Module):
    """Simplified mLSTM block - attention-based LSTM without causal convolution."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_size = hidden_size  # single block
        self.layer_norm = nn.LayerNorm(input_size)

        # Attention components
        self.Wq = SimpleBlockDiagonal(input_size, hidden_size)
        self.Wk = SimpleBlockDiagonal(input_size, hidden_size)
        self.Wv = SimpleBlockDiagonal(input_size, hidden_size)

        # LSTM gates
        self.Wi = nn.Linear(input_size, hidden_size)
        self.Wf = nn.Linear(input_size, hidden_size)
        self.Wo = nn.Linear(input_size, hidden_size)

    def forward(self, x: torch.Tensor, state: tuple) -> tuple:
        h_prev, c_prev = state
        x_norm = self.layer_norm(x)

        # Attention mechanism without convolution
        q = self.Wq(x_norm)
        k = self.Wk(x_norm) / (self.head_size**0.5)
        v = self.Wv(x)

        # LSTM with attention
        i = torch.sigmoid(self.Wi(x_norm))
        f = torch.sigmoid(self.Wf(x_norm))
        o = torch.sigmoid(self.Wo(x))

        c_t = f * c_prev + i * (v * k)
        h_t = o * torch.tanh(c_t)

        return h_t, (h_t, c_t)


class SimpleXLSTM(BaseModel):
    """Simplified xLSTM model - essentially a standard LSTM with some enhancements."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        dropout: float,
    ):
        super().__init__(model_name="SimpleXLSTM")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout = dropout

        # Store configuration
        self.model_config = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "output_dim": output_dim,
            "dropout": dropout,
        }

        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Standard LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Output layers
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the xLSTM network."""
        # Ensure input is 3D: (batch_size, sequence_length, input_dim)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence length dimension

        batch_size, seq_len, _ = x.shape

        # Project input to hidden dimension
        x = self.input_projection(x)

        # Pass through LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Take the final hidden state from the last layer
        final_h = h_n[-1]  # Shape: (batch_size, hidden_dim)

        # Apply dropout
        final_h = self.dropout_layer(final_h)

        # Project to output dimension
        output = self.output_projection(final_h)

        return torch.as_tensor(output)
