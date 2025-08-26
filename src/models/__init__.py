"""
Neural network models for GenreDiscern.
"""

from .base import BaseModel
from .neural_networks import (
    FC_model,
    CNN_model,
    LSTM_model,
    GRU_model
)
from .transformers import (
    Tr_FC,
    Tr_CNN,
    Tr_LSTM,
    Tr_GRU
)
from .xlstm import SimpleXLSTM

__all__ = [
    "BaseModel",
    "FC_model",
    "CNN_model", 
    "LSTM_model",
    "GRU_model",
    "Tr_FC",
    "Tr_CNN",
    "Tr_LSTM",
    "Tr_GRU",
    "SimpleXLSTM",
    "get_model"
]


def get_model(
    model_type: str,
    input_dim: int = None,
    hidden_dim: int = 32,
    num_layers: int = 1,
    output_dim: int = 10,
    dropout: float = 0.1,
    num_heads: int = 8,
    ff_dim: int = 128
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
    model_map = {
        'FC': lambda: FC_model(input_dim=input_dim, output_dim=output_dim, dropout=dropout),
        'CNN': lambda: CNN_model(num_classes=output_dim, dropout=dropout),
        'LSTM': lambda: LSTM_model(input_dim, hidden_dim, num_layers, output_dim, dropout),
        'GRU': lambda: GRU_model(input_dim, hidden_dim, num_layers, output_dim, dropout),
        'xLSTM': lambda: SimpleXLSTM(input_dim, hidden_dim, num_layers, output_dim, dropout),
        'Tr_FC': lambda: Tr_FC(input_dim, hidden_dim, num_layers, num_heads, ff_dim, output_dim, dropout),
        'Tr_CNN': lambda: Tr_CNN(input_dim, hidden_dim, num_layers, num_heads, ff_dim, output_dim, dropout),
        'Tr_LSTM': lambda: Tr_LSTM(input_dim, hidden_dim, num_layers, num_heads, ff_dim, output_dim, dropout),
        'Tr_GRU': lambda: Tr_GRU(input_dim, hidden_dim, num_layers, num_heads, ff_dim, output_dim, dropout)
    }
    
    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}. Available types: {list(model_map.keys())}")
    
    try:
        return model_map[model_type]()
    except Exception as e:
        raise RuntimeError(f"Failed to create model {model_type}: {e}") 