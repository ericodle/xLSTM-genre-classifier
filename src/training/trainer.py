"""
Model training module for GenreDiscern.
"""

import os
import sys
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.config import Config
from core.utils import setup_logging, get_device, ensure_directory, set_random_seed
from core.constants import (
    DEFAULT_TRAIN_SIZE,
    DEFAULT_VAL_SIZE,
    DEFAULT_TEST_SIZE,
    DEFAULT_FIGURE_WIDTH,
    DEFAULT_FIGURE_HEIGHT,
    DEFAULT_DPI,
)
from models import get_model
from data.preprocessing import AudioPreprocessor


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

        # Set random seed for reproducibility
        set_random_seed(config.training.random_seed)

        # Initialize components
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.preprocessor = AudioPreprocessor(logger)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.training_history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def setup_training(self, data_path: str, model_type: str, output_dir: str):
        """
        Setup training environment.

        Args:
            data_path: Path to training data
            model_type: Type of model to train
            output_dir: Output directory for results
        """
        self.logger.info(f"Setting up training for {model_type} model")

        # Load and preprocess data
        self._load_data(data_path)

        # Create model
        self._create_model(model_type)

        # Setup training components
        self._setup_optimizer()
        self._setup_criterion()
        self._setup_scheduler()

        # Ensure output directory exists
        ensure_directory(output_dir)
        self.output_dir = Path(output_dir)

        self.logger.info("Training setup completed")

    def _load_data(self, data_path: str):
        """Load and preprocess training data."""
        self.logger.info(f"Loading data from {data_path}")

        # Check if this is a CSV file
        if data_path.lower().endswith(".csv"):
            self._load_csv_data(data_path)
        else:
            self._load_json_data(data_path)

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
            self.logger.info(
                f"Converted {len(unique_genres)} string genres to integer labels"
            )
        else:
            raise ValueError("CSV must contain either 'label' or 'genre' column")

        # Filter out classes with too few samples (minimum 2 per class for stratified splitting)
        from collections import Counter

        label_counts = Counter(self.labels)
        min_samples_per_class = 2

        # Find classes with enough samples
        valid_classes = [
            label
            for label, count in label_counts.items()
            if count >= min_samples_per_class
        ]

        if len(valid_classes) < len(label_counts):
            # Filter data to keep only classes with enough samples
            valid_mask = np.isin(self.labels, valid_classes)
            self.features = self.features[valid_mask]
            self.labels = self.labels[valid_mask]

            # Re-encode labels to be consecutive starting from 0
            label_mapping = {
                old_label: new_label
                for new_label, old_label in enumerate(valid_classes)
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

        self.logger.info(
            f"Loaded {len(self.features)} samples with {len(np.unique(self.labels))} classes"
        )
        self.logger.info(f"Feature shape: {self.features.shape}")

    def _load_json_data(self, data_path: str):
        """Load data from JSON file."""
        # Load JSON data
        with open(data_path, "r") as f:
            data = json.load(f)

        # Check if data has the new format (features/labels arrays) or old format (file paths)
        if "features" in data and "labels" in data:
            # New format: {"features": [...], "labels": [...]}
            self.logger.info("Detected new data format with features and labels arrays")

            # Handle variable-length MFCC features by padding/truncating to consistent shape
            features_list = data["features"]
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
                    padding = [
                        [0.0] * len(feature[0])
                        for _ in range(max_frames - len(feature))
                    ]
                    padded_feature = feature + padding
                else:
                    padded_feature = feature
                padded_features.append(padded_feature)

            self.features = np.array(padded_features)
            self.labels = np.array(data["labels"])

            self.logger.info(f"Padded features to shape: {self.features.shape}")
        else:
            # Old format: {"genre/filename.wav": {"mfcc": [...]}, ...}
            self.logger.info("Detected old data format with file paths")
            features = []
            labels = []

            for file_path, file_data in data.items():
                features.append(file_data["mfcc"])
                # Extract genre from file path (assuming structure: genre/filename)
                genre = file_path.split("/")[0]
                labels.append(genre)

            # Convert to numpy arrays
            self.features = np.array(features)
            self.labels = np.array(labels)

        # Preprocess data
        self._preprocess_data()

        self.logger.info(
            f"Loaded {len(self.features)} samples with {len(np.unique(self.labels))} classes"
        )

    def _preprocess_data(self):
        """Preprocess the loaded data."""
        self.logger.info("Preprocessing data...")

        # Validate data
        if not self.preprocessor.validate_data(self.features, self.labels):
            raise ValueError("Data validation failed")

        # Encode labels
        self.encoded_labels = self.preprocessor.encode_labels(self.labels)

        # Split data
        (
            self.X_train,
            self.X_val,
            self.X_test,
            self.y_train,
            self.y_val,
            self.y_test,
        ) = self.preprocessor.split_data(
            self.features,
            self.encoded_labels,
            train_size=DEFAULT_TRAIN_SIZE,
            val_size=DEFAULT_VAL_SIZE,
            test_size=DEFAULT_TEST_SIZE,
        )

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
        self.model = get_model(
            model_type=model_type,
            input_dim=input_dim,
            hidden_dim=self.config.model.hidden_size,
            num_layers=self.config.model.num_layers,
            output_dim=num_classes,
            dropout=self.config.model.dropout,
        )

        # Move to device
        self.model = self.model.to(self.device)

        self.logger.info(
            f"Model created with {self.model.count_parameters()} parameters"
        )

    def _setup_optimizer(self):
        """Setup optimizer."""
        if self.config.model.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(
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
        """Setup loss function."""
        if self.config.model.loss_function.lower() == "crossentropy":
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(
                f"Unknown loss function: {self.config.model.loss_function}"
            )

    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        if self.config.model.lr_scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
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

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint("best_model.onnx")

            # Early stopping check
            if self._should_stop_early():
                self.logger.info("Early stopping triggered")
                break

        # Final evaluation
        test_loss, test_acc = self._evaluate_model()
        self.logger.info(f"Final test accuracy: {test_acc:.4f}")

        # Generate training plots
        self._generate_training_plots()

        return self.training_history

    def _train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
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
        elif (
            hasattr(self.model, "rnn")
            or hasattr(self.model, "lstm")
            or hasattr(self.model, "gru")
        ):
            # RNN models (LSTM, GRU) - keep 3D format (batch, time_steps, features)
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
        plt.savefig(
            plots_dir / "accuracy_plot.png", dpi=DEFAULT_DPI, bbox_inches="tight"
        )
        plt.close()

        self.logger.info(f"Training plots saved to {plots_dir}")
