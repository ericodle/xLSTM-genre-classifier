"""
Neural network models for GenreDiscern.
"""

from typing import Optional
from .base import BaseModel
from .neural_networks import FC_model, CNN_model, LSTM_model, GRU_model
from .transformers import Transformer
from .xlstm import xLSTM
from .vgg import VGG16Classifier
from src.core.constants import (
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_NUM_LAYERS,
    DEFAULT_DROPOUT,
    DEFAULT_NUM_HEADS,
    DEFAULT_FF_DIM,
    DEFAULT_CONV_LAYERS,
    DEFAULT_BASE_FILTERS,
    DEFAULT_KERNEL_SIZE,
    DEFAULT_POOL_SIZE,
    DEFAULT_FC_HIDDEN,
    DEFAULT_NUM_CLASSES,
)

__all__ = [
    "BaseModel",
    "FC_model",
    "CNN_model",
    "LSTM_model",
    "GRU_model",
    "Transformer",
    "xLSTM",
    "get_model",
]


def get_model(
    model_type: str,
    input_dim: Optional[int] = None,
    hidden_dim: int = DEFAULT_HIDDEN_SIZE,
    num_layers: int = DEFAULT_NUM_LAYERS,
    output_dim: int = DEFAULT_NUM_CLASSES,
    dropout: float = DEFAULT_DROPOUT,
    num_heads: int = DEFAULT_NUM_HEADS,
    ff_dim: int = DEFAULT_FF_DIM,
    conv_layers: int = DEFAULT_CONV_LAYERS,
    base_filters: int = DEFAULT_BASE_FILTERS,
    kernel_size: int = DEFAULT_KERNEL_SIZE,
    pool_size: int = DEFAULT_POOL_SIZE,
    fc_hidden: int = DEFAULT_FC_HIDDEN,
    block_types: Optional[list] = None,
    conv_kernel_size: int = DEFAULT_KERNEL_SIZE,
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
        "VGG16",
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
            # Use xLSTM-specific parameters
            from core.model_defaults import DEFAULTS
            xlstm_hidden = getattr(DEFAULTS, 'xlstm_hidden_size', hidden_dim)
            xlstm_layers = getattr(DEFAULTS, 'xlstm_num_layers', num_layers)
            xlstm_dropout = getattr(DEFAULTS, 'xlstm_dropout', dropout)
            return xLSTM(
                input_dim_int, xlstm_hidden, xlstm_layers, output_dim, xlstm_dropout,
                block_types=block_types, num_heads=num_heads, conv_kernel_size=conv_kernel_size
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
        elif model_type == "VGG16":
            # For VGG16, ignore input_dim; model expects (batch, 1, H, W)
            return VGG16Classifier(num_classes=output_dim, pretrained=True, dropout=dropout)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    except Exception as e:
        raise RuntimeError(f"Failed to create model {model_type}: {e}")
