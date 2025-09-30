"""
Data loading and preprocessing utilities for GenreDiscern.
"""

import json
import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional, Union
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .utils import setup_logging, ensure_directory


class AudioDataset(Dataset):
    """PyTorch Dataset for audio features."""

    def __init__(self, features: np.ndarray, labels: np.ndarray, transform=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        if self.transform:
            feature = self.transform(feature)

        return feature, label


class DataManager:
    """Manages data loading, preprocessing, and dataset creation."""

    def __init__(self, config=None, logger=None):
        self.config = config
        self.logger = logger or setup_logging()
        self.label_encoder = LabelEncoder()
        self.feature_scaler = None

    def load_json_data(
        self, json_path: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load data from JSON file containing MFCC features."""
        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            features = np.array(data["mfcc"])
            labels = np.array(data["labels"])
            mapping = data.get("mapping", [])

            self.logger.info(
                f"Loaded {len(features)} samples with {len(mapping)} classes"
            )
            self.logger.info(f"Feature shape: {features.shape}")
            self.logger.info(f"Labels shape: {labels.shape}")

            return features, labels, mapping

        except Exception as e:
            self.logger.error(f"Error loading data from {json_path}: {e}")
            raise

    def preprocess_features(self, features: np.ndarray, fit_normalizer: bool = True) -> np.ndarray:
        """Preprocess features (normalization, reshaping, etc.)."""
        # Ensure features are 2D
        if len(features.shape) == 3:
            # If features are 3D (samples, time_steps, mfcc_coeffs)
            # Reshape to 2D (samples, time_steps * mfcc_coeffs)
            features = features.reshape(features.shape[0], -1)

        if fit_normalizer:
            # Calculate normalization statistics from training data only
            self.feature_mean = np.mean(features, axis=0)
            self.feature_std = np.std(features, axis=0) + 1e-8
            self.normalizer_fitted = True

        if hasattr(self, 'normalizer_fitted') and self.normalizer_fitted:
            # Apply normalization using fitted statistics
            features = (features - self.feature_mean) / self.feature_std
        else:
            # Fallback: normalize on current data (for backward compatibility)
            features = (features - np.mean(features, axis=0)) / (
                np.std(features, axis=0) + 1e-8
            )

        return features

    def encode_labels(self, labels: np.ndarray) -> np.ndarray:
        """Encode string labels to integers."""
        if labels.dtype == object:  # String labels
            encoded_labels = self.label_encoder.fit_transform(labels)
        else:
            encoded_labels = labels

        return np.asarray(encoded_labels)

    def split_data(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train, validation, and test sets."""
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            features,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )

        # Second split: separate validation set from remaining data
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=y_temp,
        )

        self.logger.info(f"Train set: {len(X_train)} samples")
        self.logger.info(f"Validation set: {len(X_val)} samples")
        self.logger.info(f"Test set: {len(X_test)} samples")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def create_data_loaders(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create PyTorch DataLoaders for train, validation, and test sets."""

        # Create datasets
        train_dataset = AudioDataset(X_train, y_train)
        val_dataset = AudioDataset(X_val, y_val)
        test_dataset = AudioDataset(X_test, y_test)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        return train_loader, val_loader, test_loader

    def save_processed_data(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        mapping: List[str],
        output_path: str,
        filename: str,
    ) -> None:
        """Save processed data to JSON file."""
        ensure_directory(output_path)

        processed_data = {
            "mfcc": features.tolist(),
            "labels": labels.tolist(),
            "mapping": mapping,
            "metadata": {
                "num_samples": len(features),
                "num_classes": len(mapping),
                "feature_shape": features.shape,
                "preprocessed": True,
            },
        }

        output_file = os.path.join(output_path, f"{filename}.json")
        with open(output_file, "w") as f:
            json.dump(processed_data, f, indent=2)

        self.logger.info(f"Processed data saved to {output_file}")

    def get_class_distribution(
        self, labels: np.ndarray, mapping: List[str]
    ) -> Dict[str, int]:
        """Get distribution of classes in the dataset."""
        unique, counts = np.unique(labels, return_counts=True)
        distribution = {mapping[i]: count for i, count in zip(unique, counts)}
        return distribution

    def validate_data(self, features: np.ndarray, labels: np.ndarray) -> bool:
        """Validate data integrity."""
        if len(features) != len(labels):
            self.logger.error("Features and labels have different lengths")
            return False

        if len(features) == 0:
            self.logger.error("No features found")
            return False

        if np.isnan(features).any() or np.isinf(features).any():
            self.logger.error("Features contain NaN or infinite values")
            return False

        if np.isnan(labels).any() or np.isinf(labels).any():
            self.logger.error("Labels contain NaN or infinite values")
            return False

        return True
