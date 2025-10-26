"""
Comprehensive audio feature extractor for multimodal music genre classification.
Extracts different types of features optimized for different model branches.
"""

import numpy as np
import librosa
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Add src directory to path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.constants import SAMPLE_RATE, HOP_LENGTH, N_FFT


@dataclass
class MultimodalFeatures:
    """Container for different types of audio features."""
    
    # Spectral features (for CNN branch)
    mel_spectrogram: np.ndarray
    chroma: np.ndarray
    spectral_centroid: np.ndarray
    spectral_rolloff: np.ndarray
    spectral_contrast: np.ndarray
    zero_crossing_rate: np.ndarray
    
    # Temporal features (for RNN branch)
    mfcc: np.ndarray
    delta_mfcc: np.ndarray
    delta2_mfcc: np.ndarray
    
    # Statistical features (for FC branch)
    tempo: float
    beat_frames: np.ndarray
    onset_strength: np.ndarray
    harmonic_percussive_ratio: float
    spectral_bandwidth: np.ndarray
    spectral_flatness: np.ndarray


class MultimodalFeatureExtractor:
    """Extracts comprehensive audio features for multimodal classification."""
    
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        hop_length: int = HOP_LENGTH,
        n_fft: int = N_FFT,
        n_mfcc: int = 13,
        n_mels: int = 128,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the multimodal feature extractor.
        
        Args:
            sample_rate: Audio sample rate
            hop_length: Hop length for STFT
            n_fft: FFT window size
            n_mfcc: Number of MFCC coefficients
            n_mels: Number of mel bands
            logger: Logger instance
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.logger = logger or logging.getLogger(__name__)
    
    def extract_features(self, audio_path: str) -> Optional[MultimodalFeatures]:
        """
        Extract comprehensive features from an audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            MultimodalFeatures object or None if extraction fails
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Ensure minimum length
            if len(y) < self.hop_length:
                self.logger.warning(f"Audio too short: {audio_path}")
                return None
            
            # Extract spectral features (for CNN branch)
            spectral_features = self._extract_spectral_features(y, sr)
            
            # Extract temporal features (for RNN branch)
            temporal_features = self._extract_temporal_features(y, sr)
            
            # Extract statistical features (for FC branch)
            statistical_features = self._extract_statistical_features(y, sr)
            
            return MultimodalFeatures(
                # Spectral features
                mel_spectrogram=spectral_features['mel_spectrogram'],
                chroma=spectral_features['chroma'],
                spectral_centroid=spectral_features['spectral_centroid'],
                spectral_rolloff=spectral_features['spectral_rolloff'],
                spectral_contrast=spectral_features['spectral_contrast'],
                zero_crossing_rate=spectral_features['zero_crossing_rate'],
                
                # Temporal features
                mfcc=temporal_features['mfcc'],
                delta_mfcc=temporal_features['delta_mfcc'],
                delta2_mfcc=temporal_features['delta2_mfcc'],
                
                # Statistical features
                tempo=statistical_features['tempo'],
                beat_frames=statistical_features['beat_frames'],
                onset_strength=statistical_features['onset_strength'],
                harmonic_percussive_ratio=statistical_features['harmonic_percussive_ratio'],
                spectral_bandwidth=statistical_features['spectral_bandwidth'],
                spectral_flatness=statistical_features['spectral_flatness'],
            )
            
        except Exception as e:
            self.logger.error(f"Failed to extract features from {audio_path}: {e}")
            return None
    
    def _extract_spectral_features(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Extract spectral features optimized for CNN processing."""
        
        # Mel-spectrogram (main spectral representation)
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=self.n_mels, hop_length=self.hop_length, n_fft=self.n_fft
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Chroma features (harmonic content)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=self.hop_length)
        
        # Spectral centroid (brightness)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_length)
        
        # Spectral rolloff (frequency below which 85% of energy lies)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=self.hop_length)
        
        # Spectral contrast (difference between peaks and valleys)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=self.hop_length)
        
        # Zero crossing rate (percussiveness)
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)
        
        return {
            'mel_spectrogram': mel_spec_db,
            'chroma': chroma,
            'spectral_centroid': spectral_centroid,
            'spectral_rolloff': spectral_rolloff,
            'spectral_contrast': spectral_contrast,
            'zero_crossing_rate': zcr,
        }
    
    def _extract_temporal_features(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Extract temporal features optimized for RNN processing."""
        
        # MFCC features
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=self.n_mfcc, hop_length=self.hop_length, n_fft=self.n_fft
        )
        
        # Delta MFCC (first derivatives)
        delta_mfcc = librosa.feature.delta(mfcc)
        
        # Delta-delta MFCC (second derivatives)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        
        return {
            'mfcc': mfcc,
            'delta_mfcc': delta_mfcc,
            'delta2_mfcc': delta2_mfcc,
        }
    
    def _extract_statistical_features(self, y: np.ndarray, sr: int) -> Dict:
        """Extract statistical features optimized for FC processing."""
        
        # Tempo and beat tracking
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length)
        
        # Onset strength
        onset_strength = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
        
        # Harmonic vs percussive separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        harmonic_percussive_ratio = np.mean(np.abs(y_harmonic)) / (np.mean(np.abs(y_percussive)) + 1e-8)
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=self.hop_length)
        
        # Spectral flatness (noisiness)
        spectral_flatness = librosa.feature.spectral_flatness(y=y, hop_length=self.hop_length)
        
        return {
            'tempo': tempo,
            'beat_frames': beats,
            'onset_strength': onset_strength,
            'harmonic_percussive_ratio': harmonic_percussive_ratio,
            'spectral_bandwidth': spectral_bandwidth,
            'spectral_flatness': spectral_flatness,
        }
    
    def extract_features_batch(self, audio_paths: List[str]) -> List[Optional[MultimodalFeatures]]:
        """
        Extract features from multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
            
        Returns:
            List of MultimodalFeatures objects (None for failed extractions)
        """
        features_list = []
        
        for i, audio_path in enumerate(audio_paths):
            if i % 100 == 0:
                self.logger.info(f"Processing file {i+1}/{len(audio_paths)}: {audio_path}")
            
            features = self.extract_features(audio_path)
            features_list.append(features)
        
        return features_list
    
    def get_feature_shapes(self, features: MultimodalFeatures) -> Dict[str, Tuple[int, ...]]:
        """
        Get the shapes of all feature arrays.
        
        Args:
            features: MultimodalFeatures object
            
        Returns:
            Dictionary mapping feature names to their shapes
        """
        return {
            # Spectral features
            'mel_spectrogram': features.mel_spectrogram.shape,
            'chroma': features.chroma.shape,
            'spectral_centroid': features.spectral_centroid.shape,
            'spectral_rolloff': features.spectral_rolloff.shape,
            'spectral_contrast': features.spectral_contrast.shape,
            'zero_crossing_rate': features.zero_crossing_rate.shape,
            
            # Temporal features
            'mfcc': features.mfcc.shape,
            'delta_mfcc': features.delta_mfcc.shape,
            'delta2_mfcc': features.delta2_mfcc.shape,
            
            # Statistical features
            'tempo': (1,),  # Scalar
            'beat_frames': features.beat_frames.shape,
            'onset_strength': features.onset_strength.shape,
            'harmonic_percussive_ratio': (1,),  # Scalar
            'spectral_bandwidth': features.spectral_bandwidth.shape,
            'spectral_flatness': features.spectral_flatness.shape,
        }
