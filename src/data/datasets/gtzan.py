"""
GTZAN dataset handler with 10 genres.
"""

from .base import BaseDataset
import os
from typing import List, Tuple, Dict, Any


class GTZANDataset(BaseDataset):
    """GTZAN dataset handler with 10 genres."""
    
    GENRES = [
        'blues', 'classical', 'country', 'disco', 'hiphop',
        'jazz', 'metal', 'pop', 'reggae', 'rock'
    ]
    
    def __init__(self, root_dir: str):
        """
        Initialize GTZAN dataset handler.
        
        Args:
            root_dir: Root directory containing genre subdirectories
        """
        self.root_dir = root_dir
        
    def get_audio_files(self) -> List[Tuple[str, str]]:
        """
        Get all audio files with their genres from GTZAN structure.
        
        Returns:
            List of (file_path, genre) tuples
        """
        files = []
        for genre in self.GENRES:
            genre_dir = os.path.join(self.root_dir, genre)
            if os.path.exists(genre_dir):
                for file in os.listdir(genre_dir):
                    if self.validate_file(file):
                        file_path = os.path.join(genre_dir, file)
                        files.append((file_path, genre))
        return files
    
    def get_genres(self) -> List[str]:
        """
        Get list of genres in GTZAN dataset.
        
        Returns:
            List of genre names
        """
        return self.GENRES.copy()
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get dataset metadata.
        
        Returns:
            Dictionary containing dataset information
        """
        return {
            'name': 'GTZAN',
            'genres': self.GENRES,
            'genre_count': len(self.GENRES),
            'structure': 'genre-based folders',
            'description': 'GTZAN Genre Collection with 10 genres',
            'total_samples': self.get_sample_count(),
            'genre_distribution': self.get_genre_distribution()
        }
    
    def validate_file(self, file_path: str) -> bool:
        """
        Validate if a file is a valid audio file for GTZAN.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            True if file is valid, False otherwise
        """
        # GTZAN uses .wav files
        return file_path.lower().endswith('.wav')
    
    def get_genre_directory(self, genre: str) -> str:
        """
        Get the directory path for a specific genre.
        
        Args:
            genre: Genre name
            
        Returns:
            Path to genre directory
        """
        return os.path.join(self.root_dir, genre)
    
    def validate_structure(self) -> bool:
        """
        Validate that the dataset follows GTZAN structure.
        
        Returns:
            True if structure is valid, False otherwise
        """
        for genre in self.GENRES:
            genre_dir = os.path.join(self.root_dir, genre)
            if not os.path.exists(genre_dir):
                return False
            if not os.path.isdir(genre_dir):
                return False
        return True 