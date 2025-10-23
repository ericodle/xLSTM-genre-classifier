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
    batch_size: int = 32  # Reduced from 64 for better memory efficiency
    learning_rate: float = 0.00001  # Reduced from 0.001 for better stability
    max_epochs: int = 500
    weight_decay: float = 1e-4  # Increased from 1e-5 for better regularization
    optimizer: str = "adamw"  # AdamW often better for CNNs
    loss_function: str = "crossentropy"
    lr_scheduler: bool = True
    class_weight: str = "auto"  # Enable by default for imbalanced datasets
    
    # === MODEL ARCHITECTURE PARAMETERS ===
    # RNN/LSTM/GRU parameters
    hidden_size: int = 64  # Increased from 32 for better capacity
    num_layers: int = 2  # Increased from 1 for better representation
    dropout: float = 0.3  # Increased from 0.1 for better regularization
    
    # CNN parameters
    conv_layers: int = 4  # Increased for better feature extraction
    base_filters: int = 32  # Increased from 16 for more filters
    kernel_size: int = 3
    pool_size: int = 2
    fc_hidden: int = 128  # Increased from 64 for better capacity
    
    # Transformer parameters
    num_heads: int = 4  # Reduced from 8 for memory efficiency
    ff_dim: int = 64  # Reduced from 128 for memory efficiency
    
    # xLSTM parameters
    xlstm_hidden_size: int = 64  # Reduced for memory efficiency
    xlstm_num_layers: int = 2
    xlstm_dropout: float = 0.3
    
    # === TRAINING PROCESS PARAMETERS ===
    validation_split: float = 0.2
    early_stopping_patience: int = 25  # Increased from 20 for more patience
    random_seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True
    save_best_model: bool = True
    save_checkpoints: bool = True
    early_stopping: bool = True
    improvement_threshold: float = 0.0005  # More lenient for CNN models
    gradient_clip_norm: float = 1.0  # Enable gradient clipping by default
    
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
    
    def get_optimized_defaults_for_dataset(self, dataset_type: str, model_type: str = "GRU") -> Dict[str, Any]:
        """Get optimized defaults for specific datasets."""
        if dataset_type.upper() == "FMA":
            # FMA is imbalanced, use conservative settings to prevent gradient explosion
            base_fma_params = {
                "learning_rate": 0.0001,   # Much lower LR to prevent explosion
                "batch_size": 16,          # Smaller batch for stability
                "class_weight": "auto",    # Enable class weighting
                "early_stopping_patience": 30,  # Reasonable patience
                "improvement_threshold": 0.0005,  # More lenient stopping
                "weight_decay": 1e-4,      # Moderate weight decay
                "lr_scheduler": True,      # Ensure LR scheduling is on
                "gradient_clip_norm": 0.5, # Stronger gradient clipping
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
                "batch_size": 16,         # Smaller batch for memory
                "dropout": 0.1,           # Lower dropout for transformers
                "num_heads": 4,           # Fewer heads for memory efficiency
                "ff_dim": 64,             # Smaller FF dimension
                "early_stopping_patience": 25,  # Uniform patience across models
            }
        elif model_type.upper() == "CNN":
            # CNN-specific optimizations - more aggressive for FMA
            return {
                **base_params,
                "learning_rate": 0.0005,  # Lower LR for stability
                "batch_size": 16,         # Smaller batch for better gradients
                "dropout": 0.4,           # Higher dropout for regularization
                "conv_layers": 6,         # More conv layers for complex patterns
                "base_filters": 64,       # More filters for better features
                "kernel_size": 5,         # Larger kernels for MFCC patterns
                "fc_hidden": 256,         # Larger FC layer
                "early_stopping_patience": 40,  # More patience
                "weight_decay": 1e-4,     # Stronger regularization
            }
        elif model_type.upper() in ["LSTM", "GRU"]:
            # RNN-specific optimizations - conservative for stability
            return {
                **base_params,
                "learning_rate": 0.0003,  # Lower LR for stability
                "batch_size": 24,         # Moderate batch size
                "dropout": 0.25,          # Moderate dropout
                "num_layers": 2,          # Fewer layers for stability
                "hidden_size": 64,        # Moderate hidden size
                "early_stopping_patience": 30,  # Reasonable patience
                "weight_decay": 1e-4,     # Moderate weight decay
                "gradient_clip_norm": 0.5, # Stronger gradient clipping
            }
        elif model_type.upper() == "XLSTM":
            # xLSTM-specific optimizations - very conservative for numerical stability
            return {
                **base_params,
                "learning_rate": 0.0001,  # Very low LR for stability
                "batch_size": 8,          # Very small batch size for memory
                "dropout": 0.3,           # Standard dropout
                "early_stopping_patience": 40,  # More patience
                "weight_decay": 1e-4,     # Moderate weight decay
                "gradient_clip_norm": 0.5, # Same as LSTM/GRU for consistency
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
