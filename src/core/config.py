from dataclasses import dataclass
from typing import Any, Dict, Optional

from .constants import (
    DEFAULT_BASE_FILTERS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECKPOINT_INTERVAL,
    DEFAULT_CLASS_WEIGHT,
    DEFAULT_CONV_LAYERS,
    DEFAULT_DEVICE,
    DEFAULT_DROPOUT,
    DEFAULT_EARLY_STOPPING,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_FC_HIDDEN,
    DEFAULT_FF_DIM,
    DEFAULT_GRADIENT_CLIP_NORM,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_IMPROVEMENT_THRESHOLD,
    DEFAULT_INITIALIZER,
    DEFAULT_KERNEL_SIZE,
    DEFAULT_LABEL_SMOOTHING,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LOG_INTERVAL,
    DEFAULT_LOSS_FUNCTION,
    DEFAULT_LR_SCHEDULER,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_NUM_CLASSES,
    DEFAULT_NUM_HEADS,
    DEFAULT_NUM_LAYERS,
    DEFAULT_NUM_WORKERS,
    DEFAULT_OPTIMIZER,
    DEFAULT_PIN_MEMORY,
    DEFAULT_POOL_SIZE,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SAVE_BEST_MODEL,
    DEFAULT_SAVE_CHECKPOINTS,
    DEFAULT_VALIDATION_SPLIT,
    DEFAULT_WEIGHT_DECAY,
    GAN_DROPOUT,
    GAN_HIDDEN_DIM,
    GAN_LAMBDA_GP,
    GAN_LEARNING_RATE,
    GAN_N_CRITIC,
    GAN_NOISE_DIM,
    GAN_NUM_LAYERS,
)
from .model_defaults import DEFAULTS, get_defaults


@dataclass
class ModelConfig:
    """Model training configuration."""

    batch_size: int = DEFAULT_BATCH_SIZE
    hidden_size: int = DEFAULT_HIDDEN_SIZE
    num_layers: int = DEFAULT_NUM_LAYERS
    dropout: float = DEFAULT_DROPOUT
    learning_rate: float = DEFAULT_LEARNING_RATE
    max_epochs: int = DEFAULT_MAX_EPOCHS
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE
    validation_split: float = DEFAULT_VALIDATION_SPLIT
    optimizer: str = DEFAULT_OPTIMIZER
    loss_function: str = DEFAULT_LOSS_FUNCTION
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    lr_scheduler: bool = DEFAULT_LR_SCHEDULER
    class_weight: str = DEFAULT_CLASS_WEIGHT
    label_smoothing: float = DEFAULT_LABEL_SMOOTHING
    init: Optional[str] = DEFAULT_INITIALIZER

    # CNN-specific parameters
    num_classes: int = DEFAULT_NUM_CLASSES
    conv_layers: int = DEFAULT_CONV_LAYERS
    base_filters: int = DEFAULT_BASE_FILTERS
    kernel_size: int = DEFAULT_KERNEL_SIZE
    pool_size: int = DEFAULT_POOL_SIZE
    fc_hidden: int = DEFAULT_FC_HIDDEN

    # Transformer-specific parameters
    num_heads: int = DEFAULT_NUM_HEADS
    ff_dim: int = DEFAULT_FF_DIM

    # GAN-specific parameters
    gan_noise_dim: int = GAN_NOISE_DIM
    gan_hidden_dim: int = GAN_HIDDEN_DIM
    gan_num_layers: int = GAN_NUM_LAYERS
    gan_dropout: float = GAN_DROPOUT
    gan_n_critic: int = GAN_N_CRITIC
    gan_lambda_gp: float = GAN_LAMBDA_GP
    gan_learning_rate: float = GAN_LEARNING_RATE


@dataclass
class TrainingConfig:
    """Training process configuration."""

    device: str = DEFAULT_DEVICE
    num_workers: int = DEFAULT_NUM_WORKERS
    pin_memory: bool = DEFAULT_PIN_MEMORY
    save_best_model: bool = DEFAULT_SAVE_BEST_MODEL
    save_checkpoints: bool = DEFAULT_SAVE_CHECKPOINTS
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL
    log_interval: int = DEFAULT_LOG_INTERVAL
    random_seed: int = DEFAULT_RANDOM_SEED
    early_stopping: bool = DEFAULT_EARLY_STOPPING
    patience: int = DEFAULT_EARLY_STOPPING_PATIENCE
    improvement_threshold: float = DEFAULT_IMPROVEMENT_THRESHOLD
    improvement_window: int = DEFAULT_EARLY_STOPPING_PATIENCE
    gradient_clip_norm: Optional[float] = DEFAULT_GRADIENT_CLIP_NORM


class Config:

    def __init__(self):
        self.model = ModelConfig()
        self.training = TrainingConfig()

    def optimize_for_dataset(self, dataset_type: str, model_type: str = "GRU") -> None:
        """Optimize configuration for specific dataset type."""
        optimized_defaults = get_defaults(model_type, dataset_type)

        # Update model config with optimized defaults
        for key, value in optimized_defaults.items():
            if hasattr(self.model, key):
                setattr(self.model, key, value)
            elif hasattr(self.training, key):
                setattr(self.training, key, value)
