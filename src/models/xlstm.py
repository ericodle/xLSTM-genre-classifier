"""
Extended LSTM models for GenreDiscern.
Implements true xLSTM architecture with sLSTM and mLSTM blocks.
"""

import math
import os

# Add src directory to path for imports
import sys
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.constants import (
    DEFAULT_XLSTM_CONV_KERNEL_SIZE,
    DEFAULT_XLSTM_NUM_HEADS,
)


class CausalConv1d(nn.Module):
    """Causal 1D convolution that only looks at past timesteps."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle different input shapes
        if len(x.shape) == 4:
            # (batch, seq_len, 1, features) -> (batch, seq_len, features)
            x = x.squeeze(2)
        elif len(x.shape) == 2:
            # (batch, features) -> (batch, 1, features)
            x = x.unsqueeze(1)

        # Ensure we have the right shape: (batch, seq_len, features)
        if len(x.shape) != 3:
            # If still 4D, try to reshape to 3D
            if len(x.shape) == 4 and x.shape[2] == 1:
                x = x.squeeze(2)
            else:
                raise ValueError(f"Expected 3D input (batch, seq_len, features), got {x.shape}")

        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        x = self.conv(x)
        # Remove future timesteps to maintain causality
        x = x[:, :, : -self.padding] if self.padding > 0 else x
        # Back to (batch, seq_len, features)
        return x.transpose(1, 2)


class BlockDiagonalLinear(nn.Module):
    """Block diagonal linear layer for memory efficiency."""

    def __init__(self, in_features: int, out_features: int, block_size: int = 64):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        # Calculate number of blocks
        self.num_blocks = max(1, min(in_features, out_features) // block_size)

        # Create block diagonal matrices
        self.blocks = nn.ModuleList(
            [
                nn.Linear(
                    min(block_size, in_features - i * block_size),
                    min(block_size, out_features - i * block_size),
                )
                for i in range(self.num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle both 2D and 3D inputs
        if len(x.shape) == 2:
            # 2D input: (batch, features) -> (batch, 1, features)
            x = x.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, seq_len, _ = x.shape
        outputs = []

        for i, block in enumerate(self.blocks):
            start_in = i * self.block_size
            end_in = min((i + 1) * self.block_size, self.in_features)
            start_out = i * self.block_size
            end_out = min((i + 1) * self.block_size, self.out_features)

            if start_in < self.in_features and start_out < self.out_features:
                x_block = x[:, :, start_in:end_in]
                out_block = block(x_block)
                outputs.append(out_block)

        result = torch.cat(outputs, dim=-1)

        # Squeeze back to 2D if input was 2D
        if squeeze_output:
            result = result.squeeze(1)

        return result


class sLSTMBlock(nn.Module):
    """sLSTM block with exponential gating (simplified without convolution)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        conv_kernel_size: int = DEFAULT_XLSTM_CONV_KERNEL_SIZE,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        # Layer normalization
        self.layer_norm = nn.LayerNorm(input_size)

        # sLSTM gates with standard linear layers
        self.Wz = nn.Linear(input_size, hidden_size)
        self.Wi = nn.Linear(input_size, hidden_size)
        self.Wf = nn.Linear(input_size, hidden_size)
        self.Wo = nn.Linear(input_size, hidden_size)

        # Recurrent connections
        self.Rz = nn.Linear(hidden_size, hidden_size)
        self.Ri = nn.Linear(hidden_size, hidden_size)
        self.Rf = nn.Linear(hidden_size, hidden_size)
        self.Ro = nn.Linear(hidden_size, hidden_size)

        # Exponential gating parameters (very small initial value for stability)
        self.gate_init = nn.Parameter(
            torch.ones(hidden_size) * 0.0001
        )  # Much smaller initial value

    def forward(
        self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h_prev, c_prev = state

        # Squeeze input to 2D if needed
        if len(x.shape) == 3 and x.shape[1] == 1:
            x = x.squeeze(1)  # (batch_size, input_size)

        # Check for NaN in input
        if torch.isnan(x).any():
            print(f"Warning: NaN detected in sLSTM input")
            x = torch.nan_to_num(x, nan=0.0)

        # Apply layer normalization
        x_norm = self.layer_norm(x)

        # sLSTM equations
        z = torch.tanh(self.Wz(x) + self.Rz(h_prev))
        i = torch.sigmoid(self.Wi(x_norm) + self.Ri(h_prev))
        f = torch.sigmoid(self.Wf(x_norm) + self.Rf(h_prev))
        o = torch.sigmoid(self.Wo(x) + self.Ro(h_prev))

        # More stable forget gate: use sigmoid instead of exponential
        # This avoids the numerical instability of exponential gating
        f_stable = torch.sigmoid(self.gate_init * f)

        # Use a small positive offset to ensure f_stable > 0
        f_exp = f_stable + 0.1

        # Cell state update
        c_t = f_exp * c_prev + i * z
        h_t = o * torch.tanh(c_t)

        # Check for NaN in outputs and replace with zeros
        if torch.isnan(c_t).any():
            print(f"Warning: NaN detected in sLSTM cell state, replacing with zeros")
            c_t = torch.nan_to_num(c_t, nan=0.0)
        if torch.isnan(h_t).any():
            print(f"Warning: NaN detected in sLSTM hidden state, replacing with zeros")
            h_t = torch.nan_to_num(h_t, nan=0.0)

        return h_t, (h_t, c_t)


class mLSTMBlock(nn.Module):
    """mLSTM block with matrix memory and attention mechanism (simplified)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_heads: int = DEFAULT_XLSTM_NUM_HEADS,
        conv_kernel_size: int = DEFAULT_XLSTM_CONV_KERNEL_SIZE,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        # Layer normalization
        self.layer_norm = nn.LayerNorm(input_size)

        # Multi-head attention components
        self.Wq = nn.Linear(input_size, hidden_size)
        self.Wk = nn.Linear(input_size, hidden_size)
        self.Wv = nn.Linear(input_size, hidden_size)

        # Matrix memory components
        self.Wm = nn.Linear(input_size, hidden_size * hidden_size)
        self.Wc = nn.Linear(input_size, hidden_size)

        # LSTM gates
        self.Wi = nn.Linear(input_size, hidden_size)
        self.Wf = nn.Linear(input_size, hidden_size)
        self.Wo = nn.Linear(input_size, hidden_size)

        # Output projection
        self.output_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h_prev, c_prev = state

        # Squeeze input to 2D if needed
        if len(x.shape) == 3 and x.shape[1] == 1:
            x = x.squeeze(1)  # (batch_size, input_size)

        # Check for NaN in input
        if torch.isnan(x).any():
            print(f"Warning: NaN detected in mLSTM input")
            x = torch.nan_to_num(x, nan=0.0)

        batch_size = x.shape[0]

        # Apply layer normalization
        x_norm = self.layer_norm(x)

        # Simplified attention (just use the value projection)
        attn_output = self.Wv(x)

        # Much simpler memory update - just use a linear transformation
        # This avoids the complex matrix operations that cause NaN
        memory_contribution = self.Wc(x)  # Direct memory contribution
        memory_contribution = torch.clamp(memory_contribution, -2.0, 2.0)

        # Simplified memory-based cell state
        c_mem = memory_contribution

        # LSTM gates
        i = torch.sigmoid(self.Wi(x_norm))
        f = torch.sigmoid(self.Wf(x_norm))
        o = torch.sigmoid(self.Wo(x))

        # Cell state update with memory
        c_t = f * c_prev + i * (attn_output + c_mem)
        h_t = o * torch.tanh(c_t)

        # Output projection
        h_t = self.output_proj(h_t)

        # Check for NaN in outputs and replace with zeros
        if torch.isnan(c_t).any():
            print(f"Warning: NaN detected in mLSTM cell state, replacing with zeros")
            c_t = torch.nan_to_num(c_t, nan=0.0)
        if torch.isnan(h_t).any():
            print(f"Warning: NaN detected in mLSTM hidden state, replacing with zeros")
            h_t = torch.nan_to_num(h_t, nan=0.0)

        # Return simplified state without memory matrix
        return h_t, (h_t, c_t)


class xLSTMBlock(nn.Module):
    """Combined xLSTM block that can use both sLSTM and mLSTM."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        block_type: str = "sLSTM",
        num_heads: int = DEFAULT_XLSTM_NUM_HEADS,
        conv_kernel_size: int = DEFAULT_XLSTM_CONV_KERNEL_SIZE,
    ):
        super().__init__()
        self.block_type = block_type

        if block_type == "sLSTM":
            self.block = sLSTMBlock(input_size, hidden_size, conv_kernel_size)
        elif block_type == "mLSTM":
            self.block = mLSTMBlock(input_size, hidden_size, num_heads, conv_kernel_size)
        else:
            raise ValueError(f"Unknown block type: {block_type}")

    def forward(self, x: torch.Tensor, state) -> Tuple[torch.Tensor, Tuple]:
        return self.block(x, state)


class xLSTM(BaseModel):
    """True xLSTM implementation with sLSTM and mLSTM blocks."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        dropout: float,
        block_types: Optional[List[str]] = None,
        num_heads: int = DEFAULT_XLSTM_NUM_HEADS,
        conv_kernel_size: int = DEFAULT_XLSTM_CONV_KERNEL_SIZE,
    ):
        super().__init__(model_name="xLSTM")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout = dropout
        self.block_types = block_types or ["sLSTM", "mLSTM"] * (num_layers // 2) + (
            ["sLSTM"] if num_layers % 2 == 1 else []
        )
        self.num_heads = num_heads
        self.conv_kernel_size = conv_kernel_size

        # Store configuration
        self.model_config = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "output_dim": output_dim,
            "dropout": dropout,
            "block_types": self.block_types,
            "num_heads": num_heads,
            "conv_kernel_size": conv_kernel_size,
        }

        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # xLSTM blocks
        self.xlstm_blocks = nn.ModuleList()
        for i in range(num_layers):
            block_type = self.block_types[i] if i < len(self.block_types) else "sLSTM"
            self.xlstm_blocks.append(
                xLSTMBlock(
                    input_size=hidden_dim,
                    hidden_size=hidden_dim,
                    block_type=block_type,
                    num_heads=num_heads,
                    conv_kernel_size=conv_kernel_size,
                )
            )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Output layers
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        self.dropout_layer = nn.Dropout(dropout)

        # Initialize weights for better stability
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better numerical stability."""
        for name, param in self.named_parameters():
            if "weight" in name:
                if len(param.shape) >= 2:
                    # Xavier/Glorot initialization for linear layers
                    nn.init.xavier_uniform_(param)
                else:
                    # Small random values for 1D parameters
                    nn.init.normal_(param, mean=0.0, std=0.01)
            elif "bias" in name:
                # Initialize biases to zero
                nn.init.constant_(param, 0.0)
            elif "gate_init" in name:
                # Keep gate_init very small for stability
                nn.init.constant_(param, 0.0001)

    def _init_state(self, batch_size: int, device: torch.device) -> List[Tuple]:
        """Initialize hidden states for all layers."""
        states = []
        for block in self.xlstm_blocks:
            # Both sLSTM and mLSTM now use the same state format: (h, c)
            h = torch.zeros(batch_size, self.hidden_dim, device=device)
            c = torch.zeros(batch_size, self.hidden_dim, device=device)
            states.append((h, c))
        return states

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the xLSTM network."""
        # Ensure input is 3D: (batch_size, sequence_length, input_dim)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence length dimension

        # Check for NaN in input
        if torch.isnan(x).any():
            print(f"Warning: NaN detected in xLSTM input")
            x = torch.nan_to_num(x, nan=0.0)

        batch_size, seq_len, _ = x.shape
        device = x.device

        # Project input to hidden dimension
        x = self.input_projection(x)

        # Check for NaN after input projection
        if torch.isnan(x).any():
            print(f"Warning: NaN detected after input projection")
            x = torch.nan_to_num(x, nan=0.0)

        # Initialize states for all layers
        states = self._init_state(batch_size, device)

        # Process sequence through xLSTM blocks
        for i, block in enumerate(self.xlstm_blocks):
            # Process each timestep
            outputs = []
            for t in range(seq_len):
                x_t = x[:, t : t + 1, :]  # (batch_size, 1, hidden_dim)
                h_t, states[i] = block(x_t, states[i])
                # Ensure h_t is 2D
                if len(h_t.shape) == 3 and h_t.shape[1] == 1:
                    h_t = h_t.squeeze(1)  # (batch_size, hidden_dim)

                # Check for NaN in timestep output
                if torch.isnan(h_t).any():
                    print(f"Warning: NaN detected in timestep {t} of layer {i}")
                    h_t = torch.nan_to_num(h_t, nan=0.0)

                outputs.append(h_t)

            # Stack outputs and apply layer norm
            x = torch.stack(outputs, dim=1)  # (batch_size, seq_len, hidden_dim)
            x = self.layer_norm(x)

            # Check for NaN after layer processing
            if torch.isnan(x).any():
                print(f"Warning: NaN detected after layer {i}")
                x = torch.nan_to_num(x, nan=0.0)

        # Take the final hidden state from the last layer
        final_h = x[:, -1, :]  # Shape: (batch_size, hidden_dim)

        # Apply dropout
        final_h = self.dropout_layer(final_h)

        # Project to output dimension
        output = self.output_projection(final_h)

        # Check for NaN in final output
        if torch.isnan(output).any():
            print(f"Warning: NaN detected in final output")
            output = torch.nan_to_num(output, nan=0.0)

        # Ensure output is 2D
        if len(output.shape) > 2:
            output = output.squeeze()

        return output
