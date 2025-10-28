"""
Single source of truth for all model parameter defaults.
This eliminates DRY violations and ensures consistency across the codebase.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .constants import (
    DEFAULT_BASE_FILTERS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CLASS_WEIGHT,
    DEFAULT_CONV_LAYERS,
    DEFAULT_DROPOUT,
    DEFAULT_EARLY_STOPPING,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_FC_HIDDEN,
    DEFAULT_FF_DIM,
    DEFAULT_GRADIENT_CLIP_NORM,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_IMPROVEMENT_THRESHOLD,
    DEFAULT_KERNEL_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LOSS_FUNCTION,
    DEFAULT_LR_SCHEDULER,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_NUM_HEADS,
    DEFAULT_NUM_LAYERS,
    DEFAULT_NUM_WORKERS,
    DEFAULT_OPTIMIZER,
    DEFAULT_PIN_MEMORY,
    DEFAULT_POOL_SIZE,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SAVE_BEST_MODEL,
    DEFAULT_SAVE_CHECKPOINTS,
    DEFAULT_TEST_SIZE,
    DEFAULT_TRAIN_SIZE,
    DEFAULT_VAL_SIZE,
    DEFAULT_VALIDATION_SPLIT,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_XLSTM_DROPOUT,
    DEFAULT_XLSTM_HIDDEN_SIZE,
    DEFAULT_XLSTM_NUM_LAYERS,
)


@dataclass
class ModelDefaults:
    """Single source of truth for all model parameter defaults."""

    # === CORE TRAINING PARAMETERS ===
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    max_epochs: int = DEFAULT_MAX_EPOCHS
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    optimizer: str = DEFAULT_OPTIMIZER
    loss_function: str = DEFAULT_LOSS_FUNCTION
    lr_scheduler: bool = DEFAULT_LR_SCHEDULER
    class_weight: str = DEFAULT_CLASS_WEIGHT

    # === MODEL ARCHITECTURE PARAMETERS ===
    # RNN/LSTM/GRU parameters
    hidden_size: int = DEFAULT_HIDDEN_SIZE
    num_layers: int = DEFAULT_NUM_LAYERS
    dropout: float = DEFAULT_DROPOUT

    # CNN parameters
    conv_layers: int = DEFAULT_CONV_LAYERS
    base_filters: int = DEFAULT_BASE_FILTERS
    kernel_size: int = DEFAULT_KERNEL_SIZE
    pool_size: int = DEFAULT_POOL_SIZE
    fc_hidden: int = DEFAULT_FC_HIDDEN

    # Transformer parameters
    num_heads: int = DEFAULT_NUM_HEADS
    ff_dim: int = DEFAULT_FF_DIM

    # xLSTM parameters
    xlstm_hidden_size: int = DEFAULT_XLSTM_HIDDEN_SIZE
    xlstm_num_layers: int = DEFAULT_XLSTM_NUM_LAYERS
    xlstm_dropout: float = DEFAULT_XLSTM_DROPOUT

    # === TRAINING PROCESS PARAMETERS ===
    validation_split: float = DEFAULT_VALIDATION_SPLIT
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE
    random_seed: int = DEFAULT_RANDOM_SEED
    num_workers: int = DEFAULT_NUM_WORKERS
    pin_memory: bool = DEFAULT_PIN_MEMORY
    save_best_model: bool = DEFAULT_SAVE_BEST_MODEL
    save_checkpoints: bool = DEFAULT_SAVE_CHECKPOINTS
    early_stopping: bool = DEFAULT_EARLY_STOPPING
    improvement_threshold: float = DEFAULT_IMPROVEMENT_THRESHOLD
    gradient_clip_norm: float = DEFAULT_GRADIENT_CLIP_NORM

    # === DATA SPLIT PARAMETERS ===
    train_size: float = DEFAULT_TRAIN_SIZE
    val_size: float = DEFAULT_VAL_SIZE
    test_size: float = DEFAULT_TEST_SIZE

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
            "gradient_clip_norm": self.gradient_clip_norm,
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

    def get_optimized_defaults_for_dataset(
        self, dataset_type: str, model_type: str = "GRU"
    ) -> Dict[str, Any]:
        """Get optimized defaults for specific datasets."""
        if dataset_type.upper() == "FMA":
            # FMA is imbalanced, use conservative settings to prevent gradient explosion
            base_fma_params = {
                "learning_rate": 0.0001,  # Much lower LR to prevent explosion
                "batch_size": 16,  # Smaller batch for stability
                "class_weight": "auto",  # Enable class weighting
                "early_stopping_patience": 30,  # Reasonable patience
                "improvement_threshold": 0.0005,  # More lenient stopping
                "weight_decay": 1e-4,  # Moderate weight decay
                "lr_scheduler": True,  # Ensure LR scheduling is on
                "gradient_clip_norm": 0.5,  # Stronger gradient clipping
            }

            # Add model-specific FMA optimizations
            model_params = self.get_model_specific_defaults(model_type)
            return {**model_params, **base_fma_params}
        elif dataset_type.upper() == "GTZAN":
            # GTZAN is balanced, use standard settings
            return self.get_model_specific_defaults("GRU")
        else:
            # Unknown dataset, use defaults
            return self.get_model_specific_defaults("GRU")

    def get_optimized_defaults_for_model(self, model_type: str) -> Dict[str, Any]:
        """Get optimized defaults for specific model types."""
        base_params = self.get_model_specific_defaults(model_type)

        if model_type.upper() == "TRANSFORMER":
            # Transformer-specific optimizations
            return {
                **base_params,
                "learning_rate": 0.0001,  # Lower LR for transformers
                "batch_size": 16,  # Smaller batch for memory
                "dropout": 0.1,  # Lower dropout for transformers
                "num_heads": 4,  # Fewer heads for memory efficiency
                "ff_dim": 64,  # Smaller FF dimension
                "early_stopping_patience": 25,  # Uniform patience across models
            }
        elif model_type.upper() == "CNN":
            # CNN-specific optimizations - more aggressive for FMA
            return {
                **base_params,
                "learning_rate": 0.0005,  # Lower LR for stability
                "batch_size": 16,  # Smaller batch for better gradients
                "dropout": 0.4,  # Higher dropout for regularization
                "conv_layers": 6,  # More conv layers for complex patterns
                "base_filters": 64,  # More filters for better features
                "kernel_size": 5,  # Larger kernels for MFCC patterns
                "fc_hidden": 256,  # Larger FC layer
                "early_stopping_patience": 40,  # More patience
                "weight_decay": 1e-4,  # Stronger regularization
            }
        elif model_type.upper() in ["LSTM", "GRU"]:
            # RNN-specific optimizations - conservative for stability
            return {
                **base_params,
                "learning_rate": 0.0003,  # Lower LR for stability
                "batch_size": 24,  # Moderate batch size
                "dropout": 0.25,  # Moderate dropout
                "num_layers": 2,  # Fewer layers for stability
                "hidden_size": 64,  # Moderate hidden size
                "early_stopping_patience": 30,  # Reasonable patience
                "weight_decay": 1e-4,  # Moderate weight decay
                "gradient_clip_norm": 0.5,  # Stronger gradient clipping
            }
        elif model_type.upper() == "XLSTM":
            # xLSTM-specific optimizations - very conservative for numerical stability
            return {
                **base_params,
                "learning_rate": 0.0001,  # Very low LR for stability
                "batch_size": 8,  # Very small batch size for memory
                "dropout": 0.3,  # Standard dropout
                "early_stopping_patience": 40,  # More patience
                "weight_decay": 1e-4,  # Moderate weight decay
                "gradient_clip_norm": 0.5,  # Same as LSTM/GRU for consistency
            }
        else:
            return base_params


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
        # Get dataset-specific optimizations (includes model-specific params)
        return DEFAULTS.get_optimized_defaults_for_dataset(dataset_type, model_type)
    else:
        return DEFAULTS.get_optimized_defaults_for_model(model_type)


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
