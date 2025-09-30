"""
Dataset factory for creating appropriate dataset handlers.
"""

import os
from typing import Optional, Dict, Any
from .base import BaseDataset
from .gtzan import GTZANDataset
from .fma import FMADataset
import logging

logger = logging.getLogger(__name__)


class DatasetFactory:
    """Factory for creating dataset handlers."""

    @staticmethod
    def create_dataset(
        root_dir: str, dataset_type: str = "auto", **kwargs
    ) -> BaseDataset:
        """
        Create a dataset handler based on type or auto-detection.

        Args:
            root_dir: Root directory containing the dataset
            dataset_type: Dataset type ('auto', 'gtzan', 'fma')
            **kwargs: Additional arguments for specific dataset types

        Returns:
            Appropriate dataset handler instance

        Raises:
            ValueError: If dataset type cannot be determined or is invalid
        """
        if dataset_type == "auto":
            dataset_type = DatasetFactory._auto_detect_dataset(root_dir)
            logger.info(f"Auto-detected dataset type: {dataset_type}")

        if dataset_type == "gtzan":
            return GTZANDataset(root_dir)
        elif dataset_type == "fma":
            api_key = kwargs.get("api_key") or os.getenv("FMA_API_KEY")
            if not api_key:
                raise ValueError(
                    "FMA API key required. Set FMA_API_KEY environment variable "
                    "or pass api_key parameter"
                )
            tracks_csv = kwargs.get("tracks_csv")
            return FMADataset(root_dir, api_key, tracks_csv)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    @staticmethod
    def _auto_detect_dataset(root_dir: str) -> str:
        """
        Auto-detect dataset type based on directory structure.

        Args:
            root_dir: Root directory to analyze

        Returns:
            Detected dataset type ('gtzan' or 'fma')
        """
        if not os.path.exists(root_dir):
            raise ValueError(f"Directory does not exist: {root_dir}")

        # Check for GTZAN structure (genre-based folders)
        gtzan_genres = [
            "blues",
            "classical",
            "country",
            "disco",
            "hiphop",
            "jazz",
            "metal",
            "pop",
            "reggae",
            "rock",
        ]
        gtzan_count = sum(
            1 for genre in gtzan_genres if os.path.exists(os.path.join(root_dir, genre))
        )

        if gtzan_count >= 8:  # At least 8 out of 10 genres present
            return "gtzan"

        # Check for FMA structure (track_id-based folders)
        # Look for folders with 3-digit names (000, 001, etc.)
        subdirs = [
            d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
        ]
        fma_pattern_count = sum(1 for d in subdirs if d.isdigit() and len(d) == 3)

        if fma_pattern_count > 0:
            return "fma"

        # Default to GTZAN if structure is unclear
        logger.warning("Could not determine dataset type, defaulting to GTZAN")
        return "gtzan"

    @staticmethod
    def get_supported_datasets() -> Dict[str, Dict[str, Any]]:
        """
        Get information about supported dataset types.

        Returns:
            Dictionary mapping dataset names to their information
        """
        return {
            "gtzan": {
                "name": "GTZAN",
                "description": "GTZAN Genre Collection with 10 genres",
                "genres": 10,
                "file_format": ".wav",
                "structure": "genre-based folders",
                "api_required": False,
            },
            "fma": {
                "name": "FMA",
                "description": "Free Music Archive dataset with 16 genres",
                "genres": 16,
                "file_format": ".mp3",
                "structure": "track_id-based folders",
                "api_required": True,
            },
        }

    @staticmethod
    def validate_dataset(root_dir: str, dataset_type: str) -> bool:
        """
        Validate that a directory contains a valid dataset of the specified type.

        Args:
            root_dir: Root directory to validate
            dataset_type: Expected dataset type

        Returns:
            True if dataset is valid, False otherwise
        """
        try:
            dataset = DatasetFactory.create_dataset(root_dir, dataset_type)
            return (
                dataset.validate_structure()
                if hasattr(dataset, "validate_structure")
                else True
            )
        except Exception as e:
            logger.error(f"Dataset validation failed: {e}")
            return False
