"""
Transformer models for GenreDiscern.
"""

import os

# Add src directory to path for imports
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.constants import DEFAULT_TRANSFORMER_MAX_SEQ_LEN


class TransformerLayer(nn.Module):
    """Single transformer layer with self-attention and feed-forward network."""

    def __init__(
        self, input_dim: int, hidden_dim: int, num_heads: int, ff_dim: int, dropout: float
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            input_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, input_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + attn_output)

        # Feed-forward
        ff_output = self.ff(x)
        x = self.layer_norm2(x + ff_output)

        return x


class Transformer(BaseModel):
    """Transformer-only model for sequence classification."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        output_dim: int,
        dropout: float,
    ):
        super().__init__(model_name="Transformer")

        self.input_dim = input_dim  # MFCC feature dimension (13)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.output_dim = output_dim
        self.dropout = dropout

        # Store configuration
        self.model_config = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "ff_dim": ff_dim,
            "output_dim": output_dim,
            "dropout": dropout,
        }

        # Input projection to hidden_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Positional encoding (reduced for memory efficiency)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, 500, hidden_dim)
        )  # Reduced max sequence length

        # Transformer layers
        self.transformer_layers = nn.ModuleList(
            [
                TransformerLayer(hidden_dim, hidden_dim, num_heads, ff_dim, dropout)
                for _ in range(num_layers)
            ]
        )

        # Classification head
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer-only network."""
        # Input shape: (batch_size, sequence_length, input_dim)
        # For MFCC: (batch_size, 1292, 13)

        batch_size, seq_len, _ = x.shape

        # Truncate sequence if too long (memory optimization)
        max_seq_len = DEFAULT_TRANSFORMER_MAX_SEQ_LEN
        if seq_len > max_seq_len:
            x = x[:, :max_seq_len, :]
            seq_len = max_seq_len

        # Project input to hidden dimension
        x = self.input_projection(x)  # (batch_size, seq_len, hidden_dim)

        # Add positional encoding
        if seq_len <= self.pos_encoding.size(1):
            x = x + self.pos_encoding[:, :seq_len, :]
        else:
            # Handle longer sequences by truncating positional encoding
            x = x + self.pos_encoding[:, : self.pos_encoding.size(1), :]

        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        # Global average pooling and classification
        x = x.mean(dim=1)  # Average across sequence length
        x = self.classifier(x)  # Raw logits for CrossEntropyLoss

        return x
