import json
import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class DataLoaderUtils:
    """Utilities for loading and preprocessing data for evaluation."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize data loader utilities."""
        self.logger = logger or logging.getLogger(__name__)

    @staticmethod
    def load_mfcc_data(json_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load MFCC data from JSON file."""
        with open(json_path, "r") as f:
            mfcc_data = json.load(f)

        if "features" in mfcc_data and "labels" in mfcc_data:
            features_list = mfcc_data["features"]
            labels = np.array(mfcc_data["labels"])

            # Pad features to consistent length
            max_length = max(len(f) for f in features_list)
            padded_features = []
            for feature in features_list:
                if len(feature) < max_length:
                    padding_needed = max_length - len(feature)
                    padded_feature = feature + [
                        [0.0] * len(feature[0]) for _ in range(padding_needed)
                    ]
                    padded_features.append(padded_feature)
                else:
                    padded_features.append(feature)

            features = np.array(padded_features)

            # Get mapping from data if available
            if "mapping" in mfcc_data:
                mapping = mfcc_data["mapping"]
            else:
                unique_labels = sorted(list(set(labels)))
                mapping = unique_labels
                if labels.dtype == object:
                    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
                    labels = np.array([label_to_idx[label] for label in labels])

            return features, labels, mapping

        # Old format
        features = []
        labels = []
        mapping: List[str] = []
        for file_path, data_dict in mfcc_data.items():
            genre = file_path.split("/")[0]
            if genre not in mapping:
                mapping.append(genre)
            genre_idx = mapping.index(genre)
            features.append(data_dict["mfcc"])
            labels.append(genre_idx)

        features = np.array(features)
        labels = np.array(labels)
        return features, labels, mapping

    @staticmethod
    def preprocess_features(features: np.ndarray, model_type: str) -> np.ndarray:
        """Preprocess features based on model type."""
        if len(features.shape) == 3:
            if model_type == "CNN":
                return features.reshape(features.shape[0], 1, features.shape[1], features.shape[2])
            elif model_type == "RNN":
                return features
            else:  # FC
                return features.reshape(features.shape[0], -1)
        return features

    @staticmethod
    def create_dataloader(
        features: np.ndarray, labels: np.ndarray, batch_size: int = 32
    ) -> DataLoader:
        """Create a DataLoader from features and labels."""
        dataset = TensorDataset(torch.FloatTensor(features), torch.LongTensor(labels))
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    @staticmethod
    def validate_data(features: np.ndarray, labels: np.ndarray) -> bool:
        """Validate that data is properly formatted."""
        if len(features) == 0:
            return False
        if len(labels) == 0:
            return False
        if len(features) != len(labels):
            return False
        if len(np.unique(labels)) == 0:
            return False
        return True
