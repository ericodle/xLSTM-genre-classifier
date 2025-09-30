"""
Abstract base class for audio datasets.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
import os


class BaseDataset(ABC):
    """Abstract base class for audio datasets."""

    @abstractmethod
    def get_audio_files(self) -> List[Tuple[str, str]]:
        """
        Return list of (file_path, genre) tuples.

        Returns:
            List of tuples containing (audio_file_path, genre_label)
        """
        pass

    @abstractmethod
    def get_genres(self) -> List[str]:
        """
        Return list of unique genres in the dataset.

        Returns:
            List of unique genre names
        """
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Return dataset metadata.

        Returns:
            Dictionary containing dataset information
        """
        pass

    @abstractmethod
    def validate_file(self, file_path: str) -> bool:
        """
        Validate if a file is a valid audio file for this dataset.

        Args:
            file_path: Path to the audio file

        Returns:
            True if file is valid, False otherwise
        """
        pass

    def get_sample_count(self) -> int:
        """
        Get total number of audio samples in the dataset.

        Returns:
            Total number of audio files
        """
        audio_files = self.get_audio_files()
        return len(audio_files)

    def get_genre_distribution(self) -> Dict[str, int]:
        """
        Get distribution of genres in the dataset.

        Returns:
            Dictionary mapping genre names to counts
        """
        audio_files = self.get_audio_files()
        distribution = {}
        for _, genre in audio_files:
            distribution[genre] = distribution.get(genre, 0) + 1
        return distribution
