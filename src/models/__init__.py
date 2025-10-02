"""
Neural network models for GenreDiscern.
"""

from typing import Optional
from .base import BaseModel
from .neural_networks import FC_model, CNN_model, LSTM_model, GRU_model
from .transformers import Transformer
from .xlstm import SimpleXLSTM

__all__ = [
    "BaseModel",
    "FC_model",
    "CNN_model",
    "LSTM_model",
    "GRU_model",
    "Transformer",
    "SimpleXLSTM",
    "get_model",
]


def get_model(
    model_type: str,
    input_dim: Optional[int] = None,
    hidden_dim: int = 32,
    num_layers: int = 1,
    output_dim: int = 10,
    dropout: float = 0.1,
    num_heads: int = 8,
    ff_dim: int = 128,
    conv_layers: int = 3,
    base_filters: int = 16,
    kernel_size: int = 3,
    pool_size: int = 2,
    fc_hidden: int = 64,
) -> BaseModel:
    """
    Factory function to create model instances.

    Args:
        model_type: Type of model to create
        input_dim: Input dimension (required for some models)
        hidden_dim: Hidden dimension size
        num_layers: Number of layers
        output_dim: Output dimension (number of classes)
        dropout: Dropout rate
        num_heads: Number of attention heads (for transformers)
        ff_dim: Feed-forward dimension (for transformers)

    Returns:
        Model instance
    """
    # Validate model type
    valid_types = [
        "FC",
        "CNN",
        "LSTM",
        "GRU",
        "xLSTM",
        "Transformer",
    ]
    if model_type not in valid_types:
        raise ValueError(
            f"Unknown model type: {model_type}. Available types: {valid_types}"
        )

    # Ensure input_dim is not None for models that require it
    if input_dim is None:
        if model_type in [
            "FC",
            "LSTM",
            "GRU",
            "xLSTM",
            "Tr_FC",
            "Tr_CNN",
            "Tr_LSTM",
            "Tr_GRU",
        ]:
            input_dim = 13 * 100  # Default fallback for models that need it
        else:
            input_dim = 0  # Default for models that don't need it

    # At this point, input_dim is guaranteed to be int, not None
    assert input_dim is not None
    input_dim_int: int = input_dim

    try:
        if model_type == "FC":
            return FC_model(
                input_dim=input_dim_int, output_dim=output_dim, dropout=dropout
            )
        elif model_type == "CNN":
            return CNN_model(
                num_classes=output_dim, 
                dropout=dropout,
                conv_layers=conv_layers,
                base_filters=base_filters,
                kernel_size=kernel_size,
                pool_size=pool_size,
                fc_hidden=fc_hidden,
            )
        elif model_type == "LSTM":
            return LSTM_model(
                input_dim_int, hidden_dim, num_layers, output_dim, dropout
            )
        elif model_type == "GRU":
            return GRU_model(input_dim_int, hidden_dim, num_layers, output_dim, dropout)
        elif model_type == "xLSTM":
            return SimpleXLSTM(
                input_dim_int, hidden_dim, num_layers, output_dim, dropout
            )
        elif model_type == "Transformer":
            return Transformer(
                input_dim_int,
                hidden_dim,
                num_layers,
                num_heads,
                ff_dim,
                output_dim,
                dropout,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    except Exception as e:
        raise RuntimeError(f"Failed to create model {model_type}: {e}")
