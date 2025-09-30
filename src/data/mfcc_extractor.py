"""
MFCC feature extraction module.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import librosa

# Add src directory to path for imports
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.config import AudioConfig
from core.utils import (
    setup_logging,
    ensure_directory,
    is_audio_file,
    validate_audio_parameters,
)
from core.constants import AUDIO_EXTENSIONS


class MFCCExtractor:
    """Extract MFCC features from audio files."""

    def __init__(self, config: AudioConfig, logger: Optional[logging.Logger] = None):
        """
        Initialize MFCC extractor.

        Args:
            config: Audio configuration parameters
            logger: Logger instance (optional)
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # Validate configuration
        validate_audio_parameters(
            self.config.sample_rate, self.config.n_fft, self.config.hop_length
        )

    def extract_mfcc_from_file(self, audio_path: str) -> np.ndarray:
        """
        Extract MFCC features from a single audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            MFCC features as numpy array
        """
        if not is_audio_file(audio_path):
            raise ValueError(f"Invalid audio file: {audio_path}")

        try:
            # Load audio file
            self.logger.debug(f"Loading audio file: {audio_path}")
            y, sr = librosa.load(audio_path, sr=self.config.sample_rate)

            # Pad or truncate to target length
            target_length = int(self.config.sample_rate * self.config.song_length)
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)), mode="constant")
            else:
                y = y[:target_length]

            # Extract MFCC features
            mfcc = librosa.feature.mfcc(
                y=y,
                sr=sr,
                n_mfcc=self.config.mfcc_count,
                hop_length=self.config.hop_length,
                n_fft=self.config.n_fft,
            )

            # Transpose to get (time_steps, features) format
            mfcc = mfcc.T

            self.logger.debug(f"Extracted MFCC shape: {mfcc.shape}")
            return mfcc

        except Exception as e:
            self.logger.error(f"Failed to extract MFCC from {audio_path}: {e}")
            raise

    def extract_mfcc_from_directory(
        self, music_path: str, output_path: str, output_filename: str
    ) -> str:
        """
        Extract MFCC features from all audio files in a directory.

        Args:
            music_path: Path to music directory
            output_path: Path to output directory
            output_filename: Output filename (without extension)

        Returns:
            Path to output JSON file
        """
        music_path_obj = Path(music_path)
        output_path_obj = Path(output_path)

        if not music_path_obj.exists():
            raise ValueError(f"Music directory does not exist: {music_path_obj}")

        # Ensure output directory exists
        ensure_directory(output_path_obj)

        # Find all audio files
        audio_files: list[Path] = []
        for ext in AUDIO_EXTENSIONS:
            audio_files.extend(music_path_obj.rglob(f"*{ext}"))

        if not audio_files:
            raise ValueError(f"No audio files found in {music_path_obj}")

        self.logger.info(f"Found {len(audio_files)} audio files")

        # Extract features from each file
        features_data = {}
        processed_count = 0

        for audio_file in audio_files:
            try:
                # Get relative path from music directory
                rel_path = audio_file.relative_to(music_path)

                # Extract MFCC features
                mfcc_features = self.extract_mfcc_from_file(str(audio_file))

                # Store features with relative path as key
                features_data[str(rel_path)] = {
                    "mfcc": mfcc_features.tolist(),
                    "shape": mfcc_features.shape,
                    "file_path": str(audio_file),
                }

                processed_count += 1
                if processed_count % 10 == 0:
                    self.logger.info(
                        f"Processed {processed_count}/{len(audio_files)} files"
                    )

            except Exception as e:
                self.logger.warning(f"Failed to process {audio_file}: {e}")
                continue

        # Save to JSON file
        output_file = output_path_obj / f"{output_filename}.json"

        with open(output_file, "w") as f:
            json.dump(features_data, f, indent=2)

        self.logger.info(f"Successfully processed {processed_count} files")
        self.logger.info(f"Features saved to: {output_file}")

        return str(output_file)

    def get_mfcc_shape(self, audio_path: str) -> Tuple[int, int]:
        """
        Get the shape of MFCC features for a given audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (time_steps, features)
        """
        mfcc = self.extract_mfcc_from_file(audio_path)
        return mfcc.shape
