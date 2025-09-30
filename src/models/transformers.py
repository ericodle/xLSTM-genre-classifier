"""
Transformer-based models for GenreDiscern.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base import BaseModel
from .neural_networks import LSTM_model, GRU_model


class TransformerLayer(nn.Module):
    """Basic transformer layer implementation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, input_dim),
        )
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + self.dropout(attn_output))

        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))

        return x


class Tr_FC(BaseModel):
    """Transformer + Fully Connected model."""

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
        super().__init__(model_name="Tr_FC")

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

        # Output layer
        self.output_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer + FC network."""
        # Ensure input is 3D: (batch_size, sequence_length, input_dim)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension

        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        # Global average pooling and classification
        x = x.mean(dim=1)  # Average across sequence length
        x = self.output_layer(x)
        return F.log_softmax(x, dim=1)


class Tr_CNN(BaseModel):
    """Transformer + CNN model."""

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
        super().__init__(model_name="Tr_CNN")

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

        # CNN layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(3, stride=2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(3, stride=2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(1, stride=2),
            nn.BatchNorm2d(512),
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(82432, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer + CNN network."""
        # Ensure input is 3D: (batch_size, sequence_length, input_dim)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension

        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        # Add channel dimension for CNN
        x = x.unsqueeze(1)  # (batch_size, 1, sequence_length, input_dim)

        # Pass through CNN layers
        x = self.conv_layers(x)

        # Pass through FC layers
        x = self.fc_layers(x)

        return x


class Tr_LSTM(BaseModel):
    """Transformer + LSTM model."""

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
        super().__init__(model_name="Tr_LSTM")

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

        # LSTM model
        self.lstm = LSTM_model(input_dim, hidden_dim, num_layers, output_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer + LSTM network."""
        # Ensure input is 3D: (batch_size, sequence_length, input_dim)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension

        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        # Pass output of transformer to LSTM
        lstm_out = self.lstm(x)

        return F.log_softmax(lstm_out, dim=1)


class Tr_GRU(BaseModel):
    """Transformer + GRU model."""

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
        super().__init__(model_name="Tr_GRU")

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

        # GRU model
        self.gru = GRU_model(input_dim, hidden_dim, num_layers, output_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer + GRU network."""
        # Ensure input is 3D: (batch_size, sequence_length, input_dim)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension

        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        # Pass output of transformer to GRU
        gru_out = self.gru(x)

        return F.log_softmax(gru_out, dim=1)
