"""
Dataset-agnostic MFCC extractor that works with any dataset implementing BaseDataset.
"""

from typing import Dict, List, Any, Union
from .datasets.base import BaseDataset
from .datasets.factory import DatasetFactory
import librosa
import numpy as np
import json
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DatasetAgnosticMFCCExtractor:
    """MFCC extractor that works with any dataset implementing BaseDataset."""

    def __init__(self, dataset: BaseDataset, config: Dict[str, Any], logger=None):
        """
        Initialize the MFCC extractor.

        Args:
            dataset: Dataset handler implementing BaseDataset
            config: Configuration dictionary for MFCC extraction
            logger: Logger instance (optional)
        """
        self.dataset = dataset
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # Set default config values
        self.config.setdefault("sample_rate", 22050)
        self.config.setdefault("n_mfcc", 13)
        self.config.setdefault("hop_length", 512)
        self.config.setdefault("n_fft", 2048)
        self.config.setdefault("max_length", None)  # Max audio length in seconds

    def extract_mfccs(self, output_path: str) -> Dict[str, Any]:
        """
        Extract MFCCs from the dataset.

        Args:
            output_path: Path where to save the extracted features

        Returns:
            Dictionary containing the extracted features and metadata
        """
        dataset_metadata = self.dataset.get_metadata()
        self.logger.info(
            f"Starting MFCC extraction for {dataset_metadata['name']} dataset"
        )
        self.logger.info(f"Configuration: {self.config}")

        # Get all audio files
        audio_files = self.dataset.get_audio_files()
        self.logger.info(f"Found {len(audio_files)} audio files")

        if len(audio_files) == 0:
            raise ValueError("No valid audio files found in the dataset")

        # Extract features
        features = []
        labels = []
        file_paths = []
        failed_files = []

        for i, (file_path, genre) in enumerate(audio_files):
            if i % 100 == 0:
                self.logger.info(
                    f"Processing file {i+1}/{len(audio_files)}: {file_path}"
                )

            try:
                # Load audio
                audio, sr = librosa.load(file_path, sr=self.config["sample_rate"])

                # Apply length limit if specified
                if "max_duration" in self.config and self.config["max_duration"]:
                    max_samples = int(self.config["max_duration"] * sr)
                    if len(audio) > max_samples:
                        audio = audio[:max_samples]

                # Extract MFCCs using configurable parameters
                mfccs = librosa.feature.mfcc(
                    y=audio,
                    sr=sr,
                    n_mfcc=self.config["n_mfcc"],  # Use configurable n_mfcc
                    hop_length=self.config["hop_length"],
                    n_fft=self.config["n_fft"],
                )

                # Transpose to time-first format (samples, time_steps, features)
                mfccs = mfccs.T

                features.append(mfccs.tolist())
                labels.append(genre)
                file_paths.append(file_path)

            except Exception as e:
                self.logger.warning(f"Failed to process {file_path}: {e}")
                failed_files.append({"file": file_path, "error": str(e)})
                continue

        # Create output structure
        output_data = {
            "features": features,
            "labels": labels,
            "file_paths": file_paths,
            "metadata": {
                "dataset": dataset_metadata,
                "extraction_config": {
                    "sample_rate": self.config["sample_rate"],
                    "n_mfcc": self.config["n_mfcc"],
                    "n_fft": self.config["n_fft"],
                    "hop_length": self.config["hop_length"],
                    "max_duration": self.config.get("max_duration", None),
                },
                "total_samples": len(features),
                "failed_files": failed_files,
                "genres": self.dataset.get_genres(),
                "genre_distribution": self.dataset.get_genre_distribution(),
            },
        }

        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        # Log summary
        self.logger.info(f"MFCC extraction completed successfully!")
        self.logger.info(f"  - Total samples processed: {len(features)}")
        self.logger.info(f"  - Failed files: {len(failed_files)}")
        self.logger.info(f"  - Output saved to: {output_path}")
        self.logger.info(
            f"  - Feature shape: {len(features)} x {len(features[0])} x {len(features[0][0])}"
        )
        self.logger.info(f"  - MFCC coefficients: {self.config['n_mfcc']}")

        return output_data

    def validate_dataset(self) -> bool:
        """
        Validate that the dataset is properly structured.

        Returns:
            True if dataset is valid, False otherwise
        """
        try:
            # Check if we can get metadata
            metadata = self.dataset.get_metadata()
            if not metadata:
                return False

            # Check if we can get genres
            genres = self.dataset.get_genres()
            if not genres:
                return False

            # Check if we can get audio files
            audio_files = self.dataset.get_audio_files()
            if not audio_files:
                return False

            return True

        except Exception as e:
            self.logger.error(f"Dataset validation failed: {e}")
            return False

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the dataset.

        Returns:
            Dictionary containing dataset information
        """
        try:
            metadata = self.dataset.get_metadata()
            genre_dist = self.dataset.get_genre_distribution()

            return {
                "dataset_info": metadata,
                "genre_distribution": genre_dist,
                "sample_count": self.dataset.get_sample_count(),
                "supported_formats": self._get_supported_formats(),
                "validation_status": self.validate_dataset(),
            }
        except Exception as e:
            self.logger.error(f"Failed to get dataset info: {e}")
            return {"error": str(e)}

    def _get_supported_formats(self) -> List[str]:
        """Get list of supported audio file formats for this dataset."""
        if hasattr(self.dataset, "validate_file"):
            # Test common audio formats
            test_files = ["test.wav", "test.mp3", "test.flac", "test.m4a"]
            supported = []
            for test_file in test_files:
                if self.dataset.validate_file(test_file):
                    supported.append(test_file.split(".")[-1])
            return supported
        return ["unknown"]


def create_extractor(
    dataset_path: str,
    dataset_type: str = "auto",
    config: Dict[str, Any] = None,
    **kwargs,
) -> DatasetAgnosticMFCCExtractor:
    """
    Factory function to create an MFCC extractor for a dataset.

    Args:
        dataset_path: Path to the dataset directory
        dataset_type: Dataset type ('auto', 'gtzan', 'fma')
        config: MFCC extraction configuration
        **kwargs: Additional arguments for dataset creation

    Returns:
        Configured MFCC extractor instance
    """
    # Create dataset handler
    dataset = DatasetFactory.create_dataset(dataset_path, dataset_type, **kwargs)

    # Set default config if none provided
    if config is None:
        config = {
            "sample_rate": 22050,
            "n_mfcc": 13,
            "hop_length": 512,
            "n_fft": 2048,
            "max_length": 30,  # 30 seconds max
        }

    return DatasetAgnosticMFCCExtractor(dataset, config)
