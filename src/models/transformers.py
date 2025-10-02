"""
Transformer models for GenreDiscern.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseModel


class TransformerLayer(nn.Module):
    """Single transformer layer with self-attention and feed-forward network."""

    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, input_dim),
            nn.Dropout(dropout)
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

        self.input_dim = input_dim
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

        # Transformer layers
        self.transformer_layers = nn.ModuleList(
            [
                TransformerLayer(input_dim, hidden_dim, num_heads, ff_dim, dropout)
                for _ in range(num_layers)
            ]
        )

        # Simple classification head
        self.classifier = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer-only network."""
        # Ensure input is 3D: (batch_size, sequence_length, input_dim)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension

        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        # Global average pooling and classification
        x = x.mean(dim=1)  # Average across sequence length
        x = self.classifier(x)
        
        return F.softmax(x, dim=1)