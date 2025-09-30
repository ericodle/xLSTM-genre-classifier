"""
Audio preprocessing utilities for GenreDiscern.
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.utils import normalize_audio, pad_or_truncate
from core.constants import (
    DEFAULT_TRAIN_SIZE,
    DEFAULT_VAL_SIZE,
    DEFAULT_TEST_SIZE,
    DEFAULT_AUGMENTATION_FACTOR,
    DEFAULT_NOISE_STD,
    DEFAULT_FEATURE_CLIP_MIN,
    DEFAULT_FEATURE_CLIP_MAX,
    DEFAULT_EPSILON,
)


class AudioPreprocessor:
    """Handle audio and MFCC feature preprocessing."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize audio preprocessor.

        Args:
            logger: Logger instance (optional)
        """
        self.logger = logger or logging.getLogger(__name__)
        self.label_encoder = LabelEncoder()
        self.is_fitted = False

    def normalize_features(
        self, features: np.ndarray, method: str = "zscore"
    ) -> np.ndarray:
        """
        Normalize MFCC features.

        Args:
            features: Input features array
            method: Normalization method ('zscore', 'minmax', 'robust')

        Returns:
            Normalized features array
        """
        if method == "zscore":
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0)
            std[std == 0] = DEFAULT_EPSILON  # Avoid division by zero
            return np.asarray((features - mean) / std)
        elif method == "minmax":
            min_val = np.min(features, axis=0)
            max_val = np.max(features, axis=0)
            range_val = max_val - min_val
            range_val[range_val == 0] = DEFAULT_EPSILON  # Avoid division by zero
            return np.asarray((features - min_val) / range_val)
        elif method == "robust":
            median = np.median(features, axis=0)
            q75, q25 = np.percentile(features, [75, 25], axis=0)
            iqr = q75 - q25
            iqr[iqr == 0] = DEFAULT_EPSILON  # Avoid division by zero
            return np.asarray((features - median) / iqr)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def encode_labels(self, labels: List[str]) -> np.ndarray:
        """
        Encode string labels to integers.

        Args:
            labels: List of string labels

        Returns:
            Encoded labels array
        """
        if not self.is_fitted:
            self.label_encoder.fit(labels)
            self.is_fitted = True

        return np.asarray(self.label_encoder.transform(labels))

    def decode_labels(self, encoded_labels: np.ndarray) -> List[str]:
        """
        Decode integer labels back to strings.

        Args:
            encoded_labels: Encoded labels array

        Returns:
            Decoded labels list
        """
        if not self.is_fitted:
            raise ValueError("Label encoder not fitted. Call encode_labels first.")

        return list(self.label_encoder.inverse_transform(encoded_labels))

    def split_data(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        train_size: float = DEFAULT_TRAIN_SIZE,
        val_size: float = DEFAULT_VAL_SIZE,
        test_size: float = DEFAULT_TEST_SIZE,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train, validation, and test sets.

        Args:
            features: Input features
            labels: Input labels
            train_size: Proportion of training data
            val_size: Proportion of validation data
            test_size: Proportion of test data
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Validate split proportions
        total = train_size + val_size + test_size
        if abs(total - 1.0) > DEFAULT_EPSILON:
            raise ValueError(f"Split proportions must sum to 1.0, got {total}")

        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            features,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )

        # Second split: separate validation from training
        val_ratio = val_size / (train_size + val_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_ratio,
            random_state=random_state,
            stratify=y_temp,
        )

        self.logger.info(
            f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def augment_data(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        augmentation_factor: int = DEFAULT_AUGMENTATION_FACTOR,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment data using simple techniques.

        Args:
            features: Input features
            labels: Input labels
            augmentation_factor: Number of augmented samples per original sample

        Returns:
            Tuple of (augmented_features, augmented_labels)
        """
        augmented_features = []
        augmented_labels = []

        for i in range(len(features)):
            # Original sample
            augmented_features.append(features[i])
            augmented_labels.append(labels[i])

            # Augmented samples
            for _ in range(augmentation_factor - 1):
                # Add random noise
                noise = np.random.normal(0, DEFAULT_NOISE_STD, features[i].shape)
                augmented_feature = features[i] + noise

                # Ensure values are within reasonable bounds
                augmented_feature = np.clip(
                    augmented_feature,
                    DEFAULT_FEATURE_CLIP_MIN,
                    DEFAULT_FEATURE_CLIP_MAX,
                )

                augmented_features.append(augmented_feature)
                augmented_labels.append(labels[i])

        return np.array(augmented_features), np.array(augmented_labels)

    def balance_classes(
        self, features: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance classes by oversampling minority classes.

        Args:
            features: Input features
            labels: Input labels

        Returns:
            Tuple of (balanced_features, balanced_labels)
        """
        unique_labels, counts = np.unique(labels, return_counts=True)
        max_count = np.max(counts)

        balanced_features: list[list[float]] = []
        balanced_labels = []

        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            label_features = features[label_indices]

            # Add all original samples
            balanced_features.extend(label_features)
            balanced_labels.extend([label] * len(label_features))

            # Oversample if needed
            if len(label_features) < max_count:
                # Random sampling with replacement
                additional_samples = max_count - len(label_features)
                indices = np.random.choice(
                    len(label_features), additional_samples, replace=True
                )

                for idx in indices:
                    balanced_features.append(label_features[idx])
                    balanced_labels.append(label)

        return np.array(balanced_features), np.array(balanced_labels)

    def get_class_distribution(self, labels: np.ndarray) -> Dict[str, int]:
        """
        Get distribution of classes in the dataset.

        Args:
            labels: Input labels

        Returns:
            Dictionary mapping class names to counts
        """
        unique_labels, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique_labels, counts))

    def validate_data(self, features: np.ndarray, labels: np.ndarray) -> bool:
        """
        Validate input data for consistency.

        Args:
            features: Input features
            labels: Input labels

        Returns:
            True if data is valid, False otherwise
        """
        if len(features) != len(labels):
            self.logger.error(
                f"Features length ({len(features)}) != labels length ({len(labels)})"
            )
            return False

        if len(features) == 0:
            self.logger.error("Empty dataset")
            return False

        if np.isnan(features).any() or np.isinf(features).any():
            self.logger.error("Features contain NaN or infinite values")
            return False

        self.logger.info("Data validation passed")
        return True
