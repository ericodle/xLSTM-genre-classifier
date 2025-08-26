"""
FMA dataset handler with 16 genres using their API.
"""

from .base import BaseDataset
import os
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
import sys
import logging

logger = logging.getLogger(__name__)


class FMADataset(BaseDataset):
    """FMA dataset handler with 16 genres using their API."""
    
    def __init__(self, root_dir: str, api_key: str, tracks_csv: str = None):
        """
        Initialize FMA dataset handler.
        
        Args:
            root_dir: Root directory containing FMA audio files
            api_key: FMA API key for genre retrieval
            tracks_csv: Optional path to tracks CSV for faster processing
        """
        self.root_dir = root_dir
        self.api_key = api_key
        self.tracks_csv = tracks_csv
        self._fma_api = None
        self._tracks_df = None
        self._genres_df = None
        
    def _init_fma_api(self):
        """Initialize FMA API connection."""
        try:
            # Try to import FMA utils
            fma_utils_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'fma')
            if fma_utils_path not in sys.path:
                sys.path.insert(0, fma_utils_path)
            
            from utils import FreeMusicArchive
            self._fma_api = FreeMusicArchive(self.api_key)
            logger.info("FMA API initialized successfully")
        except ImportError as e:
            logger.error(f"Failed to import FMA utils: {e}")
            raise ImportError(
                "FMA utils module not found. Please ensure fma/utils.py is available "
                "in the project root or set PYTHONPATH accordingly."
            )
        except Exception as e:
            logger.error(f"Failed to initialize FMA API: {e}")
            raise
    
    def _load_tracks_data(self):
        """Load tracks metadata from CSV or API."""
        if self.tracks_csv and os.path.exists(self.tracks_csv):
            logger.info(f"Loading tracks data from CSV: {self.tracks_csv}")
            try:
                fma_utils_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'fma')
                if fma_utils_path not in sys.path:
                    sys.path.insert(0, fma_utils_path)
                
                from utils import load
                self._tracks_df = load(self.tracks_csv)
                logger.info(f"Loaded {len(self._tracks_df)} tracks from CSV")
            except Exception as e:
                logger.warning(f"Failed to load CSV, falling back to API: {e}")
                self._tracks_df = None
        
        if self._tracks_df is None:
            logger.info("Loading tracks data from FMA API (this may take a while)")
            self._init_fma_api()
            # Note: Full API loading would be implemented here
            # For now, we'll require the CSV file
            raise ValueError(
                "CSV file is required for FMA dataset. Please provide --fma-tracks-csv "
                "or ensure the file exists."
            )
    
    def get_audio_files(self) -> List[Tuple[str, str]]:
        """
        Get all audio files with their genres from FMA structure.
        
        Returns:
            List of (file_path, genre) tuples
        """
        if self._tracks_df is None:
            self._load_tracks_data()
        
        files = []
        processed = 0
        
        for track_id, track_data in self._tracks_df.iterrows():
            try:
                # Get primary genre
                genre = track_data[('track', 'genre_top')]
                if pd.isna(genre):
                    continue
                    
                # Build audio file path
                audio_path = self._get_audio_path(track_id)
                if os.path.exists(audio_path):
                    files.append((audio_path, genre))
                    processed += 1
                    
                    if processed % 1000 == 0:
                        logger.info(f"Processed {processed} tracks...")
                        
            except Exception as e:
                logger.warning(f"Failed to process track {track_id}: {e}")
                continue
        
        logger.info(f"Found {len(files)} valid audio files")
        return files
    
    def _get_audio_path(self, track_id: int) -> str:
        """Get audio file path for a track ID."""
        try:
            fma_utils_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'fma')
            if fma_utils_path not in sys.path:
                sys.path.insert(0, fma_utils_path)
            
            from utils import get_audio_path
            return get_audio_path(self.root_dir, track_id)
        except ImportError:
            # Fallback implementation if FMA utils not available
            tid_str = f'{track_id:06d}'
            return os.path.join(self.root_dir, tid_str[:3], tid_str + '.mp3')
    
    def get_genres(self) -> List[str]:
        """
        Get unique genres from the dataset.
        
        Returns:
            List of unique genre names
        """
        if self._tracks_df is None:
            self._load_tracks_data()
        
        genres = self._tracks_df[('track', 'genre_top')].dropna().unique()
        return sorted(list(genres))
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get dataset metadata.
        
        Returns:
            Dictionary containing dataset information
        """
        return {
            'name': 'FMA',
            'genres': self.get_genres(),
            'genre_count': len(self.get_genres()),
            'structure': 'track_id-based folders',
            'api_required': True,
            'description': 'Free Music Archive dataset with 16 genres',
            'total_samples': self.get_sample_count(),
            'genre_distribution': self.get_genre_distribution()
        }
    
    def validate_file(self, file_path: str) -> bool:
        """
        Validate if a file is a valid audio file for FMA.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            True if file is valid, False otherwise
        """
        # FMA uses .mp3 files
        return file_path.lower().endswith('.mp3')
    
    def get_track_info(self, track_id: int) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific track.
        
        Args:
            track_id: FMA track ID
            
        Returns:
            Dictionary with track information or None if not found
        """
        if self._tracks_df is None:
            self._load_tracks_data()
        
        if track_id in self._tracks_df.index:
            track_data = self._tracks_df.loc[track_id]
            return {
                'track_id': track_id,
                'title': track_data[('track', 'title')],
                'genre': track_data[('track', 'genre_top')],
                'artist': track_data[('artist', 'name')],
                'album': track_data[('album', 'title')]
            }
        return None 