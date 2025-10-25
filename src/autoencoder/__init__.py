"""
Autoencoder modules for raw audio feature extraction.

This package contains the fresh autoencoder approach where each song
gets its own independent autoencoder for maximum representation purity.
"""

from .song_level_autoencoder import (
    SongLevelDataset,
    SongLevelAutoencoder,
    SongLevelAutoencoderExtractor,
    extract_song_level_features
)

__all__ = [
    # Fresh song-level autoencoder (one autoencoder per song)
    'SongLevelDataset',
    'SongLevelAutoencoder',
    'SongLevelAutoencoderExtractor',
    'extract_song_level_features'
]
