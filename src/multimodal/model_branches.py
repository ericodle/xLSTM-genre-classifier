"""
Specialized model branches for multimodal music genre classification.
Each branch is optimized for specific types of audio features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple

# Add src directory to path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.base import BaseModel
from core.constants import (
    DEFAULT_HIDDEN_SIZE, DEFAULT_NUM_LAYERS, DEFAULT_DROPOUT,
    DEFAULT_CONV_LAYERS, DEFAULT_BASE_FILTERS, DEFAULT_KERNEL_SIZE,
    DEFAULT_POOL_SIZE, DEFAULT_FC_HIDDEN, DEFAULT_NUM_CLASSES
)


class SpectralCNNBranch(nn.Module):
    """
    CNN branch optimized for spectral features (mel-spectrogram, chroma, etc.).
    Processes 2D spectral representations.
    """
    
    def __init__(
        self,
        input_channels: int = 6,  # mel_spec, chroma, spectral_centroid, spectral_rolloff, spectral_contrast, zcr
        num_classes: int = DEFAULT_NUM_CLASSES,
        dropout: float = DEFAULT_DROPOUT,
        conv_layers: int = DEFAULT_CONV_LAYERS,
        base_filters: int = DEFAULT_BASE_FILTERS,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        pool_size: int = DEFAULT_POOL_SIZE,
        fc_hidden: int = DEFAULT_FC_HIDDEN,
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # Convolutional layers
        conv_layers_list = []
        in_channels = input_channels
        
        for i in range(conv_layers):
            out_channels = base_filters * (2 ** i)
            conv_layers_list.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(pool_size),
                nn.Dropout2d(dropout * 0.5)  # Less dropout for CNN
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers_list)
        
        # Calculate the size after convolutions (approximate)
        # This will be calculated dynamically in forward pass
        self.fc = nn.Sequential(
            nn.Linear(256, 128),  # Match the actual output size
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        # Store fc_hidden for dynamic calculation
        self.fc_hidden = fc_hidden
    
    def forward(self, spectral_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN branch.
        
        Args:
            spectral_features: Tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Convolutional layers
        x = self.conv_layers(spectral_features)
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc(x)
        
        return x


class TemporalRNNBranch(nn.Module):
    """
    RNN branch optimized for temporal features (MFCC, delta MFCC, etc.).
    Processes sequential temporal patterns.
    """
    
    def __init__(
        self,
        input_dim: int = 39,  # 13 MFCC + 13 delta + 13 delta2
        hidden_dim: int = DEFAULT_HIDDEN_SIZE,
        num_layers: int = DEFAULT_NUM_LAYERS,
        dropout: float = DEFAULT_DROPOUT,
        num_classes: int = DEFAULT_NUM_CLASSES,
        rnn_type: str = "GRU",  # GRU, LSTM, or RNN
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        
        # RNN layer
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0,
                bidirectional=False
            )
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0,
                bidirectional=False
            )
        else:  # RNN
            self.rnn = nn.RNN(
                input_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0,
                bidirectional=False
            )
        
        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, temporal_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through RNN branch.
        
        Args:
            temporal_features: Tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Ensure input is 3D
        if len(temporal_features.shape) == 2:
            temporal_features = temporal_features.unsqueeze(1)
        
        # RNN forward pass
        if self.rnn_type == "LSTM":
            # Initialize hidden and cell states
            h0 = torch.zeros(self.num_layers, temporal_features.size(0), self.hidden_dim).to(temporal_features.device)
            c0 = torch.zeros(self.num_layers, temporal_features.size(0), self.hidden_dim).to(temporal_features.device)
            out, _ = self.rnn(temporal_features, (h0, c0))
        else:
            # Initialize hidden state
            h0 = torch.zeros(self.num_layers, temporal_features.size(0), self.hidden_dim).to(temporal_features.device)
            out, _ = self.rnn(temporal_features, h0)
        
        # Use the last time step output
        out = out[:, -1, :]
        
        # Fully connected layers
        out = self.fc(out)
        
        return out


class StatisticalFCBranch(nn.Module):
    """
    FC branch optimized for statistical features (tempo, onset strength, etc.).
    Processes scalar and vector statistical measures.
    """
    
    def __init__(
        self,
        input_dim: int = 1000,  # Will be calculated dynamically
        hidden_dims: List[int] = None,
        dropout: float = DEFAULT_DROPOUT,
        num_classes: int = DEFAULT_NUM_CLASSES,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        # Build fully connected layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
                # Removed BatchNorm1d to avoid issues with batch size 1
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.fc_layers = nn.Sequential(*layers)
    
    def forward(self, statistical_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through FC branch.
        
        Args:
            statistical_features: Tensor of shape (batch_size, input_dim)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        return self.fc_layers(statistical_features)
    
    def update_input_dim(self, new_input_dim: int):
        """Update the input dimension and rebuild the network."""
        if new_input_dim != self.input_dim:
            self.input_dim = new_input_dim
            # Rebuild the first layer
            first_layer = self.fc_layers[0]
            new_first_layer = nn.Linear(new_input_dim, first_layer.out_features)
            # Move to the same device as the original layer
            device = next(self.parameters()).device
            new_first_layer = new_first_layer.to(device)
            self.fc_layers[0] = new_first_layer


class FeatureProcessor:
    """
    Processes raw multimodal features into tensors suitable for each branch.
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device
    
    def process_spectral_features(
        self, 
        mel_spec: np.ndarray,
        chroma: np.ndarray,
        spectral_centroid: np.ndarray,
        spectral_rolloff: np.ndarray,
        spectral_contrast: np.ndarray,
        zcr: np.ndarray
    ) -> torch.Tensor:
        """
        Process spectral features for CNN branch.
        
        Args:
            Various spectral feature arrays
            
        Returns:
            Tensor of shape (1, 6, height, time) for 6 input channels
        """
        # Stack features as separate channels (for 6 input channels)
        # All features are already (height, time) format
        features_list = [
            spectral_centroid,  # (1, time)
            spectral_rolloff,   # (1, time) 
            zcr,                 # (1, time)
            mel_spec,           # (128, time)
            chroma,             # (12, time)
            spectral_contrast   # (7, time)
        ]
        
        # Pad all features to same height (128) for stacking
        max_height = 128
        padded_features = []
        
        for f in features_list:
            if f.shape[0] < max_height:
                # Pad with zeros along the height dimension
                pad_height = max_height - f.shape[0]
                # np.pad format: ((before_height, after_height), (before_width, after_width))
                f = np.pad(f, ((0, pad_height), (0, 0)), mode='constant')
            padded_features.append(f)
        
        # Stack as channels: (6, 128, time)
        features = np.stack(padded_features, axis=0)
        
        # Pad or truncate to fixed length
        target_length = 1000
        if features.shape[2] > target_length:
            features = features[:, :, :target_length]
        else:
            pad_width = target_length - features.shape[2]
            features = np.pad(features, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
        
        # Convert to tensor and add batch dimension for 4D CNN input
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)  # (1, 6, 128, 1000)
        
        return features_tensor
    
    def process_temporal_features(
        self,
        mfcc: np.ndarray,
        delta_mfcc: np.ndarray,
        delta2_mfcc: np.ndarray
    ) -> torch.Tensor:
        """
        Process temporal features for RNN branch.
        
        Args:
            MFCC and derivative arrays
            
        Returns:
            Tensor of shape (batch_size, seq_len, 39)
        """
        # Concatenate MFCC features along feature dimension
        features = np.concatenate([mfcc, delta_mfcc, delta2_mfcc], axis=0)
        
        # Transpose to (seq_len, features)
        features = features.T
        
        # Convert to tensor and add batch dimension
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        return features_tensor
    
    def process_statistical_features(
        self,
        tempo: float,
        beat_frames: np.ndarray,
        onset_strength: np.ndarray,
        harmonic_percussive_ratio: float,
        spectral_bandwidth: np.ndarray,
        spectral_flatness: np.ndarray
    ) -> torch.Tensor:
        """
        Process statistical features for FC branch.
        
        Args:
            Various statistical features
            
        Returns:
            Tensor of shape (batch_size, feature_dim)
        """
        # Flatten all features into a single vector
        features = []
        
        # Add scalar features
        features.extend([tempo, harmonic_percussive_ratio])
        
        # Add vector features (flattened)
        features.extend(beat_frames.flatten())
        features.extend(onset_strength.flatten())
        features.extend(spectral_bandwidth.flatten())
        features.extend(spectral_flatness.flatten())
        
        # Convert to tensor and add batch dimension
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        return features_tensor
