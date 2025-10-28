"""
{Gated Recurrent Unit} model for GenreDiscern.
"""

import os

# Add src directory to path for imports
import sys
from typing import List, Optional

import torch
import torch.nn as nn

from .base import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.constants import DEFAULT_NUM_CLASSES


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

        self.fc = nn.Linear(hidden_dim, output_dim)  # Raw logits for CrossEntropyLoss

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
        out = self.fc(out[:, -1, :])  # Raw logits for CrossEntropyLoss

        return torch.as_tensor(out)
