"""
Model training module for GenreDiscern.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

# Optional import for memory-efficient JSON parsing
try:
    import ijson

    IJSON_AVAILABLE = True
except ImportError:
    IJSON_AVAILABLE = False

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt

from core.config import Config
from core.constants import (
    DEFAULT_DPI,
    DEFAULT_EPSILON,
    DEFAULT_FIGURE_HEIGHT,
    DEFAULT_FIGURE_WIDTH,
    DEFAULT_TEST_SIZE,
    DEFAULT_TRAIN_SIZE,
    DEFAULT_VAL_SIZE,
)
from core.utils import ensure_directory, get_device, set_random_seed, setup_logging
from models import get_model
from models.base import BaseModel
from training.losses import FocalLoss, LabelSmoothingCrossEntropyLoss


class ModelTrainer:
    """Manage the entire model training lifecycle."""

    def __init__(self, config: Config, logger: Optional[logging.Logger] = None):
        """
        Initialize model trainer.

        Args:
            config: Configuration object
            logger: Logger instance (optional)
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.device = get_device()
        self.logger.info(f"Using device: {self.device}")

        # Set random seed for reproducibility
        set_random_seed(config.training.random_seed)

        # Initialize components
        self.model: Optional[BaseModel] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion: Optional[torch.nn.Module] = None
        self.scheduler = None
        # Initialize preprocessor for normalization and label encoding
        self.preprocessor = None  # Will be initialized with LabelEncoder
        self.feature_mean = None
        self.feature_std = None
        self.normalizer_fitted = False
        self.tb_writer: Optional[SummaryWriter] = None

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.training_history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def setup_training(
        self,
        data_path: str,
        model_type: str,
        output_dir: str,
        max_samples: int = None,
        memory_efficient: bool = False,
    ):
        """
        Setup training environment.

        Args:
            data_path: Path to training data
            model_type: Type of model to train
            output_dir: Output directory for results
            max_samples: Maximum number of samples to use (for memory optimization)
            memory_efficient: Use memory-efficient data loading
        """
        self.logger.info(f"Setting up training for {model_type} model")

        # Load and preprocess data
        self._load_data(data_path, max_samples, memory_efficient)

        # Create model
        self._create_model(model_type)

        # Setup training components
        self._setup_optimizer()
        self._setup_criterion()
        self._setup_scheduler()

        # Ensure output directory exists
        ensure_directory(output_dir)
        self.output_dir = Path(output_dir)

        # Initialize TensorBoard writer
        try:
            self.tb_writer = SummaryWriter(log_dir=str(self.output_dir))
            self.tb_writer.add_text("info", "Training started", 0)
        except Exception:
            self.tb_writer = None

        self.logger.info("Training setup completed")

        # Log model graph and a sample input image to TensorBoard (best-effort)
        try:
            sample_batch = next(iter(self.train_loader))[0].to(self.device)
            # Log graph
            if self.tb_writer is not None:
                try:
                    self.tb_writer.add_graph(self.model, sample_batch)
                except Exception:
                    pass
            # Log MFCC heatmap of first sample if 2D/3D input
            if self.tb_writer is not None and sample_batch.dim() in (2, 3):
                try:
                    # Expect (batch, time, feats) or (batch, feats)
                    sample = sample_batch[0]
                    if sample.dim() == 2:
                        mfcc = sample.detach().cpu().numpy().T  # (feats, time)
                    else:
                        mfcc = sample.detach().cpu().numpy()[None, :]  # fallback
                    import io

                    import matplotlib.pyplot as plt
                    from PIL import Image

                    fig, ax = plt.subplots()
                    cax = ax.imshow(mfcc, aspect="auto", origin="lower")
                    plt.colorbar(cax)
                    ax.set_title("Sample MFCC")
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png")
                    buf.seek(0)
                    image = Image.open(buf)
                    import torchvision.transforms as transforms

                    image_tensor = transforms.ToTensor()(image)
                    self.tb_writer.add_image("Input/MFCC_Sample0", image_tensor, 0)
                    plt.close(fig)
                    buf.close()
                except Exception:
                    pass
        except Exception:
            pass

    def _load_data(self, data_path: str, max_samples: int = None, memory_efficient: bool = False):
        """Load and preprocess training data."""
        self.logger.info(f"Loading data from {data_path}")

        # Check if data_path is a directory with pre-split data (train.json, val.json, test.json)
        if os.path.isdir(data_path):
            train_json = os.path.join(data_path, "train.json")
            val_json = os.path.join(data_path, "val.json")
            test_json = os.path.join(data_path, "test.json")

            if all(os.path.exists(f) for f in [train_json, val_json, test_json]):
                self.logger.info(
                    "Detected pre-split data directory (train.json, val.json, test.json)"
                )
                self.data_is_presplit = True
                self._load_presplit_json_data(train_json, val_json, test_json, max_samples)
                return

        # If we got here, it's not a valid pre-split directory
        raise ValueError(
            f"Expected a directory containing train.json, val.json, and test.json files.\n"
            f"Got: {data_path}\n"
            f"Use src/data/MFCC_GTZAN_extract.py to create pre-split data before training."
        )

    def _load_presplit_json_data(
        self, train_json: str, val_json: str, test_json: str, max_samples: int = None
    ):
        """Load pre-split data from three separate JSON files (train, val, test)."""
        self.logger.info("Loading pre-split data from train.json, val.json, test.json")

        def load_split_file(json_path: str, split_name: str):
            """Helper to load a single split file."""
            with open(json_path, "r") as f:
                data = json.load(f)

            features = data.get("features", [])
            labels = data.get("labels", [])

            # Extract mapping if available
            if "mapping" in data and not hasattr(self, "genre_names"):
                self.genre_names = data["mapping"]
                self.logger.info(f"Found genre mapping: {self.genre_names}")

            # Limit samples if specified
            if max_samples and len(features) > max_samples:
                self.logger.info(
                    f"Limiting {split_name} to {max_samples} samples (from {len(features)})"
                )
                features = features[:max_samples]
                labels = labels[:max_samples]

            # Pad features to consistent shape
            max_frames = max(len(feature) for feature in features)
            min_frames = min(len(feature) for feature in features)

            self.logger.info(
                f"{split_name}: {len(features)} samples, frame range: {min_frames}-{max_frames}"
            )

            # Pad all features to the same length
            padded_features = []
            for feature in features:
                if len(feature) < max_frames:
                    padding = [[0.0] * len(feature[0]) for _ in range(max_frames - len(feature))]
                    padded_feature = feature + padding
                else:
                    padded_feature = feature
                padded_features.append(padded_feature)

            return np.array(padded_features, dtype=np.float32), np.array(labels)

        # Load all three splits
        X_train, y_train = load_split_file(train_json, "Training")
        X_val, y_val = load_split_file(val_json, "Validation")
        X_test, y_test = load_split_file(test_json, "Test")

        self.logger.info(f"Loaded pre-split data:")
        self.logger.info(f"  Train: {len(X_train)} samples")
        self.logger.info(f"  Val:   {len(X_val)} samples")
        self.logger.info(f"  Test:  {len(X_test)} samples")

        # Set attributes for training
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

        # Note: features and labels are not set for pre-split data

        # Preprocess data after loading
        self._preprocess_data()

    def _load_csv_data(self, data_path: str):
        """Load data from CSV file with pre-extracted MFCC features."""
        self.logger.info("Detected CSV format with pre-extracted MFCC features")

        import pandas as pd

        # Load CSV data with error handling for encoding issues
        try:
            df = pd.read_csv(data_path, encoding="utf-8")
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(data_path, encoding="latin-1")
            except UnicodeDecodeError:
                df = pd.read_csv(data_path, encoding="cp1252")

        # Extract MFCC features (columns starting with 'mfcc')
        mfcc_columns = [col for col in df.columns if col.startswith("mfcc")]
        self.logger.info(f"Found {len(mfcc_columns)} MFCC features: {mfcc_columns}")

        # Extract features and labels
        self.features = df[mfcc_columns].values.astype(np.float32)

        # Check if we have a 'label' column (integer) or 'genre' column (string)
        if "label" in df.columns:
            self.labels = df["label"].values
            self.logger.info("Using 'label' column for target values")
        elif "genre" in df.columns:
            # Convert string genres to integer labels
            unique_genres = sorted(df["genre"].unique())
            genre_to_label = {genre: idx for idx, genre in enumerate(unique_genres)}
            self.labels = np.array([genre_to_label[genre] for genre in df["genre"]])
            self.logger.info(f"Converted {len(unique_genres)} string genres to integer labels")
        else:
            raise ValueError("CSV must contain either 'label' or 'genre' column")

        # Filter out classes with too few samples (minimum 2 per class for stratified splitting)
        from collections import Counter

        label_counts = Counter(self.labels)
        min_samples_per_class = 2

        # Find classes with enough samples
        valid_classes = [
            label for label, count in label_counts.items() if count >= min_samples_per_class
        ]

        if len(valid_classes) < len(label_counts):
            # Filter data to keep only classes with enough samples
            labels_array = self.labels.values if hasattr(self.labels, "values") else self.labels
            labels_array = np.asarray(labels_array)
            valid_mask = np.isin(labels_array, valid_classes)
            self.features = self.features[valid_mask]
            self.labels = self.labels[valid_mask]

            # Re-encode labels to be consecutive starting from 0
            label_mapping = {
                old_label: new_label for new_label, old_label in enumerate(valid_classes)
            }
            self.labels = np.array([label_mapping[label] for label in self.labels])

            self.logger.info(
                f"Filtered to {len(valid_classes)} classes with at least {min_samples_per_class} samples each"
            )
            self.logger.info(
                f"Removed {len(label_counts) - len(valid_classes)} classes with insufficient samples"
            )

        # Preprocess data
        self._preprocess_data()

        labels_array = self.labels.values if hasattr(self.labels, "values") else self.labels
        labels_array = np.asarray(labels_array)
        self.logger.info(
            f"Loaded {len(self.features)} samples with {len(np.unique(labels_array))} classes"
        )
        self.logger.info(f"Feature shape: {self.features.shape}")

    def _load_json_data(
        self, data_path: str, max_samples: int = None, memory_efficient: bool = False
    ):
        """Load data from JSON file with memory optimization options."""
        self.logger.info(f"Loading JSON data from {data_path}")

        if memory_efficient:
            self.logger.info("Using memory-efficient loading")
            self._load_json_data_memory_efficient(data_path, max_samples)
        else:
            self._load_json_data_standard(data_path, max_samples)

    def _load_json_data_standard(self, data_path: str, max_samples: int = None):
        """Standard JSON data loading (loads entire file into memory)."""
        # Load JSON data
        with open(data_path, "r") as f:
            data = json.load(f)

        # Check if data has the new format (features/labels arrays) or old format (file paths)
        if "features" in data and "labels" in data:
            # New format: {"features": [...], "labels": [...]}
            self.logger.info("Detected new data format with features and labels arrays")

            # Extract genre mapping if available
            if "mapping" in data:
                self.genre_names = data["mapping"]
                self.logger.info(f"Found genre mapping: {self.genre_names}")
            else:
                # Fallback: create generic genre names
                unique_labels = sorted(set(data["labels"]))
                self.genre_names = [f"Genre_{i}" for i in unique_labels]
                self.logger.warning(
                    f"No genre mapping found, using generic names: {self.genre_names}"
                )

            # Limit samples if specified
            features_list = data["features"]
            labels_list = data["labels"]

            if max_samples and len(features_list) > max_samples:
                self.logger.info(f"Limiting to {max_samples} samples (from {len(features_list)})")
                features_list = features_list[:max_samples]
                labels_list = labels_list[:max_samples]

            # Handle variable-length MFCC features by padding/truncating to consistent shape
            max_frames = max(len(feature) for feature in features_list)
            min_frames = min(len(feature) for feature in features_list)

            self.logger.info(
                f"MFCC features: {len(features_list)} samples, frame range: {min_frames}-{max_frames}"
            )

            # Pad all features to the same length (max_frames)
            padded_features = []
            for feature in features_list:
                if len(feature) < max_frames:
                    # Pad with zeros
                    padding = [[0.0] * len(feature[0]) for _ in range(max_frames - len(feature))]
                    padded_feature = feature + padding
                else:
                    padded_feature = feature
                padded_features.append(padded_feature)

            self.features = np.array(padded_features, dtype=np.float32)
            self.labels = np.array(labels_list)  # Keep as strings, will be encoded in preprocessing

            self.logger.info(f"Padded features to shape: {self.features.shape}")

            # Preprocess data after loading
            self._preprocess_data()
        else:
            # Old format: {"genre/filename.wav": {"mfcc": [...]}, ...}
            self.logger.info("Detected old data format with file paths")

    def _load_json_data_memory_efficient(self, data_path: str, max_samples: int = None):
        """Memory-efficient JSON data loading using streaming."""
        if not IJSON_AVAILABLE:
            self.logger.warning("ijson not available, falling back to standard loading")
            self._load_json_data_standard(data_path, max_samples)
            return

        self.logger.info("Using streaming JSON parser for memory efficiency")

        features_list = []
        labels_list = []

        # Stream parse the JSON file
        with open(data_path, "rb") as f:
            # Parse features array
            features_parser = ijson.items(f, "features.item")
            for i, feature in enumerate(features_parser):
                if max_samples and i >= max_samples:
                    break
                features_list.append(feature)
                if i % 1000 == 0:
                    self.logger.info(f"Loaded {i} features...")

        # Reset file pointer and parse labels
        with open(data_path, "rb") as f:
            labels_parser = ijson.items(f, "labels.item")
            for i, label in enumerate(labels_parser):
                if max_samples and i >= max_samples:
                    break
                labels_list.append(label)

        self.logger.info(f"Loaded {len(features_list)} samples using streaming parser")

        # Extract genre mapping if available (need to reload the file for this)
        try:
            with open(data_path, "r") as f:
                data = json.load(f)
                if "mapping" in data:
                    self.genre_names = data["mapping"]
                    self.logger.info(f"Found genre mapping: {self.genre_names}")
                else:
                    # Fallback: create generic genre names
                    unique_labels = sorted(set(labels_list))
                    self.genre_names = [f"Genre_{i}" for i in unique_labels]
                    self.logger.warning(
                        f"No genre mapping found, using generic names: {self.genre_names}"
                    )
        except Exception as e:
            self.logger.warning(f"Could not load genre mapping: {e}")
            unique_labels = sorted(set(labels_list))
            self.genre_names = [f"Genre_{i}" for i in unique_labels]

        # Handle variable-length MFCC features by padding/truncating to consistent shape
        max_frames = max(len(feature) for feature in features_list)
        min_frames = min(len(feature) for feature in features_list)

        self.logger.info(
            f"MFCC features: {len(features_list)} samples, frame range: {min_frames}-{max_frames}"
        )

        # Pad all features to the same length (max_frames)
        padded_features = []
        for feature in features_list:
            if len(feature) < max_frames:
                # Pad with zeros
                padding = [[0.0] * len(feature[0]) for _ in range(max_frames - len(feature))]
                padded_feature = feature + padding
            else:
                padded_feature = feature
            padded_features.append(padded_feature)

        self.features = np.array(padded_features, dtype=np.float32)
        self.labels = np.array(labels_list)  # Keep as strings, will be encoded in preprocessing

        self.logger.info(f"Padded features to shape: {self.features.shape}")

        # Preprocess data after loading
        self._preprocess_data()

    def _preprocess_data(self):
        """Preprocess the loaded data."""
        self.logger.info("Preprocessing data...")

        # Fit label encoder on training labels, then encode all splits
        self.logger.info("Fitting label encoder and encoding labels...")
        # Fit encoder on training data
        from sklearn.preprocessing import LabelEncoder

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.y_train)
        # Transform all splits
        self.y_train = self.label_encoder.transform(self.y_train)
        self.y_val = self.label_encoder.transform(self.y_val)
        self.y_test = self.label_encoder.transform(self.y_test)

        # Normalize features properly (fit on training data only)
        self.logger.info("Normalizing features...")
        # Fit normalizer on training data
        self.feature_mean = np.mean(self.X_train, axis=0)
        self.feature_std = np.std(self.X_train, axis=0)
        self.feature_std[self.feature_std == 0] = DEFAULT_EPSILON
        self.normalizer_fitted = True

        # Apply normalization using fitted statistics
        self.X_train = (self.X_train - self.feature_mean) / self.feature_std
        self.X_val = (self.X_val - self.feature_mean) / self.feature_std
        self.X_test = (self.X_test - self.feature_mean) / self.feature_std

        # Create data loaders
        self._create_data_loaders()

    def _create_data_loaders(self):
        """Create PyTorch data loaders."""
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(self.X_train)
        y_train_tensor = torch.LongTensor(self.y_train)
        X_val_tensor = torch.FloatTensor(self.X_val)
        y_val_tensor = torch.LongTensor(self.y_val)
        X_test_tensor = torch.FloatTensor(self.X_test)
        y_test_tensor = torch.LongTensor(self.y_test)

        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.config.model.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.config.model.batch_size, shuffle=False
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.config.model.batch_size, shuffle=False
        )

    def _create_model(self, model_type: str):
        """Create the specified model."""
        self.logger.info(f"Creating {model_type} model")

        # Get input shape from data
        input_shape = self.X_train.shape[1:]
        num_classes = len(np.unique(self.y_train))

        # Calculate input dimension based on model type
        if model_type == "FC":
            # For FC models, we need the total flattened size
            if len(input_shape) == 1:  # (features,) - CSV format
                input_dim = input_shape[0]  # features = 20 for CSV
            elif len(input_shape) == 2:  # (time_steps, features) - JSON format
                input_dim = input_shape[0] * input_shape[1]  # time_steps * features
            else:
                input_dim = np.prod(input_shape)  # multiply all dimensions
        else:
            # For other models (CNN, LSTM, etc.), use the feature dimension
            if len(input_shape) == 1:  # (features,) - CSV format
                input_dim = input_shape[0]  # features = 20 for CSV
            elif len(input_shape) == 2:  # (time_steps, features) - JSON format
                input_dim = input_shape[1]  # features = 13
            else:
                input_dim = input_shape[0]

        self.logger.info(f"Input shape: {input_shape}, Input dim: {input_dim}")

        # Create model
        if model_type == "CNN":
            # For CNN, pass architecture parameters
            self.model = get_model(
                model_type=model_type,
                input_dim=input_dim,
                hidden_dim=self.config.model.hidden_size,
                num_layers=self.config.model.num_layers,
                output_dim=num_classes,
                dropout=self.config.model.dropout,
                conv_layers=getattr(self.config.model, "conv_layers", 3),
                base_filters=getattr(self.config.model, "base_filters", 16),
                kernel_size=getattr(self.config.model, "kernel_size", 3),
                pool_size=getattr(self.config.model, "pool_size", 2),
                fc_hidden=getattr(self.config.model, "fc_hidden", 64),
            )
        else:
            # For other models, use standard parameters
            kwargs = {
                "model_type": model_type,
                "input_dim": input_dim,
                "hidden_dim": self.config.model.hidden_size,
                "num_layers": self.config.model.num_layers,
                "output_dim": num_classes,
                "dropout": self.config.model.dropout,
            }

            # For VGG16 and ViT, pass the number of MFCC features and pretrained flag
            if model_type in ["VGG16", "ViT"] and len(input_shape) == 2:
                # input_shape is (time_steps, features)
                kwargs["num_mfcc_features"] = input_shape[1]  # number of MFCC coefficients
                # Get pretrained flag from config
                pretrained = getattr(self.config.model, "pretrained", True)
                kwargs["pretrained"] = pretrained
                self.logger.info(
                    f"{model_type}: Using {input_shape[1]} MFCC features, pretrained={pretrained}"
                )

            self.model = get_model(**kwargs)

        # Move to device
        self.model = self.model.to(self.device)

        self.logger.info(f"Model created with {self.model.count_parameters()} parameters")

        # Optional initializer (no-op by default)
        init_scheme = getattr(self.config.model, "init", None)
        if init_scheme:
            self._apply_initializer(init_scheme)
            self.logger.info(f"Applied initializer: {init_scheme}")

    def _apply_initializer(self, init_scheme: str) -> None:
        import torch.nn.init as init

        def init_linear(m: torch.nn.Module):
            if isinstance(m, torch.nn.Linear):
                if init_scheme == "xavier":
                    init.xavier_uniform_(m.weight)
                elif init_scheme == "kaiming":
                    init.kaiming_uniform_(m.weight, nonlinearity="relu")
                else:
                    init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

        def init_conv(m: torch.nn.Module):
            if isinstance(m, torch.nn.Conv2d):
                if init_scheme == "kaiming":
                    init.kaiming_normal_(m.weight, nonlinearity="relu")
                else:
                    init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

        def init_rnn(m: torch.nn.Module):
            if isinstance(m, (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN)):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        init.xavier_uniform_(param)
                    elif "weight_hh" in name:
                        if init_scheme in ("orthogonal", "rnn"):
                            init.orthogonal_(param)
                        else:
                            init.xavier_uniform_(param)
                    elif "bias" in name:
                        init.zeros_(param)
                        # LSTM forget gate bias to 1
                        if isinstance(m, torch.nn.LSTM) and param.shape[0] % 4 == 0:
                            hidden = param.shape[0] // 4
                            param.data[hidden : 2 * hidden] = 1.0

        # Apply by module type
        self.model.apply(init_linear)
        self.model.apply(init_conv)
        self.model.apply(init_rnn)

    def _setup_optimizer(self):
        """Setup optimizer."""
        if self.config.model.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.model.learning_rate,
                weight_decay=self.config.model.weight_decay,
            )
        elif self.config.model.optimizer.lower() == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.model.learning_rate,
                weight_decay=self.config.model.weight_decay,
            )
        elif self.config.model.optimizer.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.model.learning_rate,
                momentum=0.9,
                weight_decay=self.config.model.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.model.optimizer}")

    def _setup_criterion(self):
        """Setup loss function with optional class weighting."""
        if self.config.model.loss_function.lower() == "crossentropy":
            # Handle class weighting
            class_weights = None
            if self.config.model.class_weight != "none":
                if self.config.model.class_weight == "auto":
                    # Calculate class weights automatically
                    if hasattr(self, "labels") and self.labels is not None:
                        labels_array = (
                            self.labels.values if hasattr(self.labels, "values") else self.labels
                        )
                        labels_array = np.asarray(labels_array)
                        class_sample_count = np.bincount(labels_array.astype(int))
                        class_weights = 1.0 / (class_sample_count + 1e-8)
                        class_weights = (
                            class_weights / class_weights.sum() * len(class_sample_count)
                        )
                        class_weights = torch.tensor(
                            class_weights, dtype=torch.float32, device=self.device
                        )
                        self.logger.info(f"Using automatic class weights: {class_weights.tolist()}")
                else:
                    # Parse comma-separated weights
                    try:
                        class_weights = torch.tensor(
                            [float(x) for x in self.config.model.class_weight.split(",")],
                            dtype=torch.float32,
                            device=self.device,
                        )
                        self.logger.info(f"Using manual class weights: {class_weights.tolist()}")
                    except ValueError as e:
                        self.logger.warning(f"Failed to parse class weights: {e}")
                        class_weights = None

            # Check if we should use label smoothing
            label_smoothing = getattr(self.config.model, "label_smoothing", 0.0)
            if label_smoothing > 0:
                num_classes = len(np.unique(self.y_train))
                self.criterion = LabelSmoothingCrossEntropyLoss(
                    smoothing=label_smoothing, num_classes=num_classes
                )
                self.logger.info(f"Using label smoothing with smoothing={label_smoothing}")
            else:
                self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        elif self.config.model.loss_function.lower() == "focal":
            # Use Focal Loss for imbalanced datasets
            self.criterion = FocalLoss(alpha=1.0, gamma=2.0)
            self.logger.info("Using Focal Loss for imbalanced dataset")
        else:
            raise ValueError(f"Unknown loss function: {self.config.model.loss_function}")

    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        if self.config.model.lr_scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=5
            )

    def train(self) -> Dict:
        """Execute the training loop with dynamic early stopping."""
        self.logger.info("Starting training...")
        self.logger.info(
            f"Early stopping: Will stop if validation accuracy doesn't improve by {self.config.training.improvement_threshold:.1%} over {self.config.training.improvement_window} epochs"
        )

        # Use configurable maximum epoch count
        max_epochs = self.config.model.max_epochs

        for epoch in range(max_epochs):
            self.current_epoch = epoch + 1

            # Training phase
            train_loss, train_acc = self._train_epoch()

            # Validation phase
            val_loss, val_acc = self._validate_epoch()

            # Update learning rate
            if self.scheduler:
                self.scheduler.step(val_loss)

            # Log progress
            self.logger.info(
                f"Epoch {self.current_epoch}/{max_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

            # Save training history
            self.training_history["train_loss"].append(train_loss)
            self.training_history["train_acc"].append(train_acc)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["val_acc"].append(val_acc)

            # TensorBoard scalars
            if self.tb_writer is not None:
                try:
                    self.tb_writer.add_scalar("Loss/Train", train_loss, self.current_epoch)
                    self.tb_writer.add_scalar("Loss/Validation", val_loss, self.current_epoch)
                    self.tb_writer.add_scalar("Accuracy/Train", train_acc, self.current_epoch)
                    self.tb_writer.add_scalar("Accuracy/Validation", val_acc, self.current_epoch)
                    if self.optimizer is not None and len(self.optimizer.param_groups) > 0:
                        self.tb_writer.add_scalar(
                            "LearningRate",
                            self.optimizer.param_groups[0].get("lr", 0.0),
                            self.current_epoch,
                        )
                    # Weights histograms (per parameter)
                    for name, param in self.model.named_parameters():
                        try:
                            self.tb_writer.add_histogram(
                                f"Weights/{name}", param, self.current_epoch
                            )
                        except Exception:
                            pass
                    # Gradient histograms using collected values (best-effort)
                    grad_vals = getattr(self, "_last_gradient_values", None)
                    if isinstance(grad_vals, dict):
                        for gname, chunks in grad_vals.items():
                            if not chunks:
                                continue
                            try:
                                flat = torch.cat(chunks)
                                self.tb_writer.add_histogram(
                                    f"Gradients/{gname}", flat, self.current_epoch
                                )
                            except Exception:
                                pass
                except Exception:
                    pass

            # Save best model based on validation accuracy (for classification)
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss  # Also track the loss at best accuracy
                self._save_checkpoint("best_model.onnx")
                self.logger.info(
                    f"New best model saved! Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}"
                )

            # Early stopping check
            if self._should_stop_early():
                self.logger.info("Early stopping triggered")
                break

        # Final evaluation
        test_loss, test_acc = self._evaluate_model()
        self.logger.info(f"Final test accuracy: {test_acc:.4f}")

        # Generate training plots
        self._generate_training_plots()

        # Export final model to ONNX
        self._export_to_onnx()

        # Close TensorBoard writer
        if self.tb_writer is not None:
            try:
                # Log hparams summary
                try:
                    hparams = {
                        "batch_size": self.config.model.batch_size,
                        "learning_rate": self.config.model.learning_rate,
                        "hidden_size": getattr(self.config.model, "hidden_size", None),
                        "num_layers": getattr(self.config.model, "num_layers", None),
                        "dropout": self.config.model.dropout,
                        "optimizer": self.config.model.optimizer,
                    }
                    metrics = {
                        "final_test_accuracy": float(
                            self.best_val_acc if hasattr(self, "best_val_acc") else 0.0
                        ),
                        "final_test_loss": float(
                            self.best_val_loss if hasattr(self, "best_val_loss") else 0.0
                        ),
                    }
                    # Filter None keys
                    hparams = {k: v for k, v in hparams.items() if v is not None}
                    self.tb_writer.add_hparams(hparams, metrics)
                except Exception:
                    pass
                self.tb_writer.add_text("info", "Training finished", self.current_epoch)
                self.tb_writer.close()
            except Exception:
                pass

        return self.training_history

    def _train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        if self.model is None or self.optimizer is None or self.criterion is None:
            raise RuntimeError("Model, optimizer, or criterion not initialized")

        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()

            # Apply gradient clipping if specified
            if (
                hasattr(self.config.training, "gradient_clip_norm")
                and self.config.training.gradient_clip_norm
            ):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.gradient_clip_norm
                )

            self.optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def _validate_epoch(self) -> Tuple[float, float]:
        """Validate for one epoch."""
        if self.model is None or self.criterion is None:
            raise RuntimeError("Model or criterion not initialized")

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def _evaluate_model(self) -> Tuple[float, float]:
        """Evaluate model on test set."""
        if self.model is None or self.criterion is None:
            raise RuntimeError("Model or criterion not initialized")

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        avg_loss = total_loss / len(self.test_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def _should_stop_early(self) -> bool:
        """Check if early stopping should be triggered based on accuracy improvement."""
        if not self.config.training.early_stopping:
            return False

        improvement_window = self.config.training.improvement_window
        improvement_threshold = self.config.training.improvement_threshold

        # Need at least the improvement window epochs to check for improvement
        if len(self.training_history["val_acc"]) < improvement_window:
            return False

        # Check if validation accuracy hasn't improved by more than threshold over the window
        recent_accuracies = self.training_history["val_acc"][-improvement_window:]

        # Calculate improvement from the beginning of the window to now
        improvement = recent_accuracies[-1] - recent_accuracies[0]

        # Stop if improvement is less than the threshold
        if improvement < improvement_threshold:
            self.logger.info(
                f"Early stopping triggered: accuracy improved by only {improvement:.4f} over last {improvement_window} epochs (threshold: {improvement_threshold:.4f})"
            )
            return True

        return False

    def _save_checkpoint(self, filename: str):
        """Save model as ONNX format."""
        if self.model is None or self.optimizer is None:
            raise RuntimeError("Model or optimizer not initialized")

        # Convert .pth to .onnx
        if filename.endswith(".pth"):
            filename = filename.replace(".pth", ".onnx")

        onnx_path = self.output_dir / filename

        # Update model metadata
        self.model.training_history = self.training_history
        self.model.is_trained = True

        # Get input shape from sample data
        sample_data = next(iter(self.train_loader))[0]
        input_shape = self._get_input_shape(sample_data)

        # Save as ONNX using BaseModel method
        # Temporarily move model to CPU for ONNX export
        original_device = next(self.model.parameters()).device
        self.model.cpu()

        try:
            self.model.save_model(
                str(onnx_path),
                input_shape,
                save_optimizer=True,
                optimizer_state=self.optimizer.state_dict(),
            )
        finally:
            # Move model back to original device
            self.model.to(original_device)

        # Save additional training metadata
        def convert_config_to_dict(config):
            """Convert config object to serializable dictionary."""
            if hasattr(config, "__dict__"):
                result = {}
                for key, value in config.__dict__.items():
                    if hasattr(value, "__dict__"):
                        result[key] = convert_config_to_dict(value)
                    else:
                        result[key] = (
                            str(value)
                            if not isinstance(
                                value, (str, int, float, bool, list, dict, type(None))
                            )
                            else value
                        )
                return result
            else:
                return str(config)

        training_metadata = {
            "epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss,
            "config": convert_config_to_dict(self.config),
        }

        metadata_path = onnx_path.parent / (onnx_path.stem + "_training_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(training_metadata, f, indent=2)

        self.logger.info(f"Model saved as ONNX: {onnx_path}")

    def _get_input_shape(self, sample_data):
        """Get input shape for ONNX export based on model type."""
        if hasattr(self.model, "conv_layers"):
            # CNN model - reshape to 4D (batch, channels, height, width)
            if len(sample_data.shape) == 3:
                # (batch, time_steps, features) -> (1, time_steps, features)
                return (sample_data.shape[1], sample_data.shape[2])
            else:
                return sample_data.shape[1:]
        # VGG or ViT style models (no conv_layers attribute)
        elif (
            hasattr(self.model, "vgg")
            or hasattr(self.model, "vit")
            or getattr(self.model, "model_name", "").upper() in ["VGG16", "VIT"]
        ):
            if len(sample_data.shape) == 3:
                # Provide (time, features); model will add channel internally
                return (sample_data.shape[1], sample_data.shape[2])
            elif len(sample_data.shape) == 4:
                # Already (batch, C, H, W) or (batch, H, W); drop batch/channel as needed
                # Prefer (H, W)
                if sample_data.shape[1] in (1, 3):
                    return (sample_data.shape[2], sample_data.shape[3])
                else:
                    return (sample_data.shape[1], sample_data.shape[2])
        elif (
            hasattr(self.model, "rnn") or hasattr(self.model, "lstm") or hasattr(self.model, "gru")
        ):
            # RNN models (LSTM, GRU) - keep 3D format (batch, time_steps, features)
            return sample_data.shape[1:]
        elif hasattr(self.model, "transformer_layers"):
            # Transformer model - keep 3D format (batch, time_steps, features)
            return sample_data.shape[1:]
        elif (
            hasattr(self.model, "xlstm_blocks")
            or getattr(self.model, "model_name", "").upper() == "XLSTM"
        ):
            # xLSTM model - keep 3D format (batch, time_steps, features)
            return sample_data.shape[1:]
        else:
            # FC model - flatten to 2D (batch, features)
            if len(sample_data.shape) == 3:
                return (sample_data.shape[1] * sample_data.shape[2],)
            else:
                return sample_data.shape[1:]

    def _export_to_onnx(self):
        """Export model to ONNX format - now handled by _save_checkpoint."""
        # This method is now redundant since _save_checkpoint handles ONNX export
        # Keeping for backward compatibility but delegating to _save_checkpoint
        self._save_checkpoint("model.onnx")

    def _generate_training_plots(self):
        """Generate training visualization plots."""
        self.logger.info("Generating training plots...")

        # Create plots directory
        plots_dir = self.output_dir / "training_plots"
        plots_dir.mkdir(exist_ok=True)

        # Loss plot
        plt.figure(figsize=(DEFAULT_FIGURE_WIDTH, DEFAULT_FIGURE_HEIGHT))
        plt.plot(self.training_history["train_loss"], label="Train Loss")
        plt.plot(self.training_history["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(plots_dir / "loss_plot.png", dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close()

        # Accuracy plot
        plt.figure(figsize=(DEFAULT_FIGURE_WIDTH, DEFAULT_FIGURE_HEIGHT))
        plt.plot(self.training_history["train_acc"], label="Train Accuracy")
        plt.plot(self.training_history["val_acc"], label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig(plots_dir / "accuracy_plot.png", dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Training plots saved to {plots_dir}")
