"""
Single source of truth for all model parameter defaults.
This eliminates DRY violations and ensures consistency across the codebase.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class ModelDefaults:
    """Single source of truth for all model parameter defaults."""
    
    # === CORE TRAINING PARAMETERS ===
    batch_size: int = 64
    learning_rate: float = 0.01
    max_epochs: int = 500
    weight_decay: float = 1e-5
    optimizer: str = "adam"
    loss_function: str = "crossentropy"
    lr_scheduler: bool = True
    class_weight: str = "none"
    
    # === MODEL ARCHITECTURE PARAMETERS ===
    # RNN/LSTM/GRU parameters
    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.1
    
    # CNN parameters
    conv_layers: int = 3
    base_filters: int = 16
    kernel_size: int = 3
    pool_size: int = 2
    fc_hidden: int = 64
    
    # Transformer parameters
    num_heads: int = 8
    ff_dim: int = 128
    
    # === TRAINING PROCESS PARAMETERS ===
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    random_seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True
    save_best_model: bool = True
    save_checkpoints: bool = True
    early_stopping: bool = True
    
    # === DATA SPLIT PARAMETERS ===
    train_size: float = 0.7
    val_size: float = 0.15
    test_size: float = 0.15
    
    def get_model_specific_defaults(self, model_type: str) -> Dict[str, Any]:
        """Get model-specific parameter defaults."""
        base_params = {
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "weight_decay": self.weight_decay,
            "optimizer": self.optimizer,
            "loss_function": self.loss_function,
            "lr_scheduler": self.lr_scheduler,
            "class_weight": self.class_weight,
            "validation_split": self.validation_split,
            "early_stopping_patience": self.early_stopping_patience,
            "random_seed": self.random_seed,
        }
        
        if model_type.upper() in ["LSTM", "GRU", "RNN"]:
            return {
                **base_params,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
            }
        elif model_type.upper() == "CNN":
            return {
                **base_params,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "conv_layers": self.conv_layers,
                "base_filters": self.base_filters,
                "kernel_size": self.kernel_size,
                "pool_size": self.pool_size,
                "fc_hidden": self.fc_hidden,
            }
        elif model_type.upper() in ["TRANSFORMER", "TR_FC", "TR_CNN", "TR_LSTM", "TR_GRU"]:
            return {
                **base_params,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
            }
        elif model_type.upper() == "FC":
            return {
                **base_params,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
            }
        else:
            # Default to RNN parameters for unknown models
            return {
                **base_params,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
            }
    
    def get_optimized_defaults_for_dataset(self, dataset_type: str) -> Dict[str, Any]:
        """Get optimized defaults for specific datasets."""
        if dataset_type.upper() == "FMA":
            # FMA is imbalanced, so use more conservative settings
            return {
                **self.get_model_specific_defaults("GRU"),
                "learning_rate": 0.0001,  # Lower LR for stability
                "batch_size": 16,         # Smaller batch for better gradients
                "dropout": 0.4,           # Higher dropout for regularization
                "num_layers": 3,          # More layers for complex patterns
                "class_weight": "auto",   # Enable class weighting
                "early_stopping_patience": 5,  # More patience for imbalanced data
            }
        elif dataset_type.upper() == "GTZAN":
            # GTZAN is balanced, use standard settings
            return self.get_model_specific_defaults("GRU")
        else:
            # Unknown dataset, use defaults
            return self.get_model_specific_defaults("GRU")


# Global instance - single source of truth
DEFAULTS = ModelDefaults()


def get_defaults(model_type: str = "GRU", dataset_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Get parameter defaults for a model type and optional dataset.
    
    Args:
        model_type: Type of model (GRU, LSTM, CNN, etc.)
        dataset_type: Type of dataset (FMA, GTZAN, etc.) for optimized defaults
        
    Returns:
        Dictionary of parameter defaults
    """
    if dataset_type:
        return DEFAULTS.get_optimized_defaults_for_dataset(dataset_type)
    else:
        return DEFAULTS.get_model_specific_defaults(model_type)


def update_defaults(**kwargs) -> None:
    """
    Update global defaults (useful for testing or configuration).
    
    Args:
        **kwargs: Parameter names and values to update
    """
    for key, value in kwargs.items():
        if hasattr(DEFAULTS, key):
            setattr(DEFAULTS, key, value)
        else:
            raise ValueError(f"Unknown parameter: {key}")


# Convenience functions for common use cases
def get_fma_defaults(model_type: str = "GRU") -> Dict[str, Any]:
    """Get optimized defaults for FMA dataset."""
    return get_defaults(model_type, "FMA")


def get_gtzan_defaults(model_type: str = "GRU") -> Dict[str, Any]:
    """Get optimized defaults for GTZAN dataset."""
    return get_defaults(model_type, "GTZAN")
