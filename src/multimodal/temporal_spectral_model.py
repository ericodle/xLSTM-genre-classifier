"""
Simplified multimodal model using only temporal + spectral branches.
This model combines the best performing branches (temporal: 50.67%, spectral: 37.33%)
while excluding the poor performing statistical branch (10.00%).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional

# Add src directory to path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.base import BaseModel
from core.constants import DEFAULT_NUM_CLASSES, DEFAULT_DROPOUT
from .model_branches import SpectralCNNBranch, TemporalRNNBranch, FeatureProcessor


class TemporalSpectralFusion(nn.Module):
    """
    Fusion mechanism for temporal + spectral branches.
    Uses learnable weighted combination.
    """
    
    def __init__(
        self,
        input_dims: List[int],
        num_classes: int = DEFAULT_NUM_CLASSES,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        
        # Learnable weights for each branch
        self.branch_weights = nn.Parameter(torch.ones(2) / 2)
        
        # Optional fusion network for refined combination
        total_input_dim = sum(input_dims)
        self.fusion_network = nn.Sequential(
            nn.Linear(total_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, branch_outputs: List[torch.Tensor], method: str = "learned") -> torch.Tensor:
        """
        Forward pass with temporal+spectral fusion.
        
        Args:
            branch_outputs: List of [spectral_output, temporal_output]
            method: Fusion method ("weighted", "concat", or "learned")
            
        Returns:
            Fused output logits
        """
        if method == "weighted":
            # Simple weighted average
            weights = F.softmax(self.branch_weights, dim=0)
            fused_output = (
                weights[0] * branch_outputs[0] +
                weights[1] * branch_outputs[1]
            )
            return fused_output
        elif method == "concat":
            # Concatenate and pass through fusion network
            concatenated = torch.cat(branch_outputs, dim=-1)
            fused_output = self.fusion_network(concatenated)
            return fused_output
        else:  # "learned" - adaptive combination
            # Use concatenation fusion for better learning
            concatenated = torch.cat(branch_outputs, dim=-1)
            fused_output = self.fusion_network(concatenated)
            return fused_output


class TemporalSpectralModel(BaseModel):
    """
    Simplified multimodal model using only temporal + spectral branches.
    """
    
    def __init__(
        self,
        num_classes: int = DEFAULT_NUM_CLASSES,
        fusion_method: str = "learned",
        dropout: float = DEFAULT_DROPOUT,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.device = device
        
        # Individual branches
        self.spectral_cnn = SpectralCNNBranch(
            input_channels=6,
            num_classes=num_classes,
            dropout=dropout
        )
        
        self.temporal_rnn = TemporalRNNBranch(
            input_dim=39,  # 13 MFCC + 13 delta + 13 delta2
            num_classes=num_classes,
            dropout=dropout
        )
        
        # Fusion mechanism
        if fusion_method == "weighted":
            # Simple weighted average
            self.branch_weights = nn.Parameter(torch.ones(2) / 2)
            self.fusion_method = "weighted"
        else:
            # Learned concatenation fusion
            input_dims = [num_classes, num_classes]  # Both branches output num_classes
            self.fusion = TemporalSpectralFusion(
                input_dims=input_dims,
                num_classes=num_classes,
                dropout=dropout
            )
            self.fusion_method = "learned"
        
        # Feature processor
        self.feature_processor = FeatureProcessor(device=device)
        
        # Move to device
        self.to(device)
    
    def forward(self, multimodal_features) -> torch.Tensor:
        """
        Forward pass through temporal+spectral model.
        
        Args:
            multimodal_features: MultimodalFeatures object or dict
            
        Returns:
            Output logits
        """
        # Process features for each branch
        spectral_tensor = self.feature_processor.process_spectral_features(
            multimodal_features.mel_spectrogram,
            multimodal_features.chroma,
            multimodal_features.spectral_centroid,
            multimodal_features.spectral_rolloff,
            multimodal_features.spectral_contrast,
            multimodal_features.zero_crossing_rate
        )
        
        temporal_tensor = self.feature_processor.process_temporal_features(
            multimodal_features.mfcc,
            multimodal_features.delta_mfcc,
            multimodal_features.delta2_mfcc
        )
        
        # Forward pass through branches
        spectral_output = self.spectral_cnn(spectral_tensor)
        temporal_output = self.temporal_rnn(temporal_tensor)
        
        # Fuse branch outputs
        if self.fusion_method == "weighted":
            # Simple weighted average
            weights = F.softmax(self.branch_weights, dim=0)
            fused_output = (
                weights[0] * spectral_output +
                weights[1] * temporal_output
            )
        else:
            # Use fusion network
            branch_outputs = [spectral_output, temporal_output]
            fused_output = self.fusion(branch_outputs, method="learned")
        
        return fused_output
    
    def get_branch_outputs(self, multimodal_features) -> Dict[str, torch.Tensor]:
        """
        Get individual branch outputs for analysis.
        
        Args:
            multimodal_features: MultimodalFeatures object
            
        Returns:
            Dictionary with branch outputs
        """
        # Process features
        spectral_tensor = self.feature_processor.process_spectral_features(
            multimodal_features.mel_spectrogram,
            multimodal_features.chroma,
            multimodal_features.spectral_centroid,
            multimodal_features.spectral_rolloff,
            multimodal_features.spectral_contrast,
            multimodal_features.zero_crossing_rate
        )
        
        temporal_tensor = self.feature_processor.process_temporal_features(
            multimodal_features.mfcc,
            multimodal_features.delta_mfcc,
            multimodal_features.delta2_mfcc
        )
        
        # Get branch outputs
        spectral_output = self.spectral_cnn(spectral_tensor)
        temporal_output = self.temporal_rnn(temporal_tensor)
        
        return {
            "spectral": spectral_output,
            "temporal": temporal_output
        }

