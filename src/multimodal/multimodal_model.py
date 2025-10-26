"""
Fusion mechanisms and main multimodal model for music genre classification.
Combines outputs from specialized branches using various fusion strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

# Add src directory to path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.base import BaseModel
from core.constants import DEFAULT_NUM_CLASSES, DEFAULT_DROPOUT
from .model_branches import SpectralCNNBranch, TemporalRNNBranch, StatisticalFCBranch, FeatureProcessor


class AttentionFusion(nn.Module):
    """
    Attention-based fusion mechanism for combining branch outputs.
    Learns to weight different branches based on their relevance.
    """
    
    def __init__(
        self,
        input_dims: List[int],
        hidden_dim: int = 128,
        num_classes: int = DEFAULT_NUM_CLASSES,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        
        self.num_branches = len(input_dims)
        self.hidden_dim = hidden_dim
        
        # Project each branch output to common dimension
        self.branch_projections = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) for input_dim in input_dims
        ])
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * self.num_branches, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_branches),
            nn.Softmax(dim=-1)
        )
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, branch_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass with attention-based fusion.
        
        Args:
            branch_outputs: List of tensors from different branches
            
        Returns:
            Fused output logits
        """
        # Project branch outputs to common dimension
        projected_outputs = []
        for i, output in enumerate(branch_outputs):
            projected = self.branch_projections[i](output)
            projected_outputs.append(projected)
        
        # Concatenate for attention computation
        concatenated = torch.cat(projected_outputs, dim=-1)
        
        # Compute attention weights
        attention_weights = self.attention(concatenated)
        
        # Weighted combination
        fused_output = torch.zeros_like(projected_outputs[0])
        for i, output in enumerate(projected_outputs):
            fused_output += attention_weights[:, i:i+1] * output
        
        # Final classification
        logits = self.classifier(fused_output)
        
        return logits


class ConcatFusion(nn.Module):
    """
    Simple concatenation-based fusion mechanism.
    """
    
    def __init__(
        self,
        input_dims: List[int],
        hidden_dim: int = 256,
        num_classes: int = DEFAULT_NUM_CLASSES,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        
        total_input_dim = sum(input_dims)
        
        self.fusion_network = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, branch_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass with concatenation-based fusion.
        
        Args:
            branch_outputs: List of tensors from different branches
            
        Returns:
            Fused output logits
        """
        # Concatenate all branch outputs
        concatenated = torch.cat(branch_outputs, dim=-1)
        
        # Pass through fusion network
        logits = self.fusion_network(concatenated)
        
        return logits


class MultimodalModel(BaseModel):
    """
    Main multimodal model that combines specialized branches.
    """
    
    def __init__(
        self,
        num_classes: int = DEFAULT_NUM_CLASSES,
        fusion_method: str = "attention",  # "attention", "concat", or "weighted"
        dropout: float = DEFAULT_DROPOUT,
        device: str = "cpu",
        # Branch-specific parameters
        cnn_params: Optional[Dict] = None,
        rnn_params: Optional[Dict] = None,
        fc_params: Optional[Dict] = None,
    ):
        super().__init__(model_name="MultimodalModel")
        
        self.num_classes = num_classes
        self.fusion_method = fusion_method
        self.dropout = dropout
        self.device = device
        
        # Default branch parameters
        if cnn_params is None:
            cnn_params = {"input_channels": 6, "dropout": dropout}  # 6 channels: centroid, rolloff, zcr, mel_spec, chroma, contrast
        if rnn_params is None:
            rnn_params = {"input_dim": 39, "dropout": dropout}
        if fc_params is None:
            fc_params = {"input_dim": 1000, "dropout": dropout}
        
        # Initialize branches
        self.spectral_cnn = SpectralCNNBranch(
            num_classes=num_classes,
            **cnn_params
        )
        
        self.temporal_rnn = TemporalRNNBranch(
            num_classes=num_classes,
            **rnn_params
        )
        
        # Calculate statistical features input dimension dynamically
        # We'll initialize this after the first forward pass
        self.statistical_fc = None
        self._statistical_input_dim = None
        
        # Initialize fusion mechanism
        if fusion_method == "attention":
            self.fusion = AttentionFusion(
                input_dims=[num_classes, num_classes, num_classes],
                num_classes=num_classes,
                dropout=dropout
            )
        elif fusion_method == "concat":
            self.fusion = ConcatFusion(
                input_dims=[num_classes, num_classes, num_classes],
                num_classes=num_classes,
                dropout=dropout
            )
        elif fusion_method == "weighted":
            # Simple weighted average
            self.branch_weights = nn.Parameter(torch.ones(3) / 3)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Feature processor
        self.feature_processor = FeatureProcessor(device=device)
        
        # Move to device
        self.to(device)
    
    def forward(self, multimodal_features) -> torch.Tensor:
        """
        Forward pass through multimodal model.
        
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
        
        statistical_tensor = self.feature_processor.process_statistical_features(
            multimodal_features.tempo,
            multimodal_features.beat_frames,
            multimodal_features.onset_strength,
            multimodal_features.harmonic_percussive_ratio,
            multimodal_features.spectral_bandwidth,
            multimodal_features.spectral_flatness
        )
        
        # Initialize statistical FC branch if not done yet
        if self.statistical_fc is None:
            input_dim = statistical_tensor.shape[1]
            self.statistical_fc = StatisticalFCBranch(
                input_dim=input_dim,
                num_classes=self.num_classes,
                dropout=self.dropout
            ).to(self.device)
        else:
            # Update input dimension if it changed
            current_input_dim = statistical_tensor.shape[1]
            if current_input_dim != self.statistical_fc.input_dim:
                self.statistical_fc.update_input_dim(current_input_dim)
        
        # Forward pass through branches
        spectral_output = self.spectral_cnn(spectral_tensor)
        temporal_output = self.temporal_rnn(temporal_tensor)
        statistical_output = self.statistical_fc(statistical_tensor)
        
        # Fuse branch outputs
        if self.fusion_method == "weighted":
            # Simple weighted average
            weights = F.softmax(self.branch_weights, dim=0)
            fused_output = (
                weights[0] * spectral_output +
                weights[1] * temporal_output +
                weights[2] * statistical_output
            )
        else:
            # Use fusion network
            branch_outputs = [spectral_output, temporal_output, statistical_output]
            fused_output = self.fusion(branch_outputs)
        
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
        
        statistical_tensor = self.feature_processor.process_statistical_features(
            multimodal_features.tempo,
            multimodal_features.beat_frames,
            multimodal_features.onset_strength,
            multimodal_features.harmonic_percussive_ratio,
            multimodal_features.spectral_bandwidth,
            multimodal_features.spectral_flatness
        )
        
        # Get branch outputs
        spectral_output = self.spectral_cnn(spectral_tensor)
        temporal_output = self.temporal_rnn(temporal_tensor)
        statistical_output = self.statistical_fc(statistical_tensor)
        
        return {
            "spectral": spectral_output,
            "temporal": temporal_output,
            "statistical": statistical_output
        }
    
    def get_attention_weights(self, multimodal_features) -> Optional[torch.Tensor]:
        """
        Get attention weights if using attention fusion.
        
        Args:
            multimodal_features: MultimodalFeatures object
            
        Returns:
            Attention weights tensor or None
        """
        if self.fusion_method != "attention":
            return None
        
        # Process features and get branch outputs
        branch_outputs = self.get_branch_outputs(multimodal_features)
        
        # Get attention weights from fusion layer
        branch_outputs_list = list(branch_outputs.values())
        
        # Project branch outputs
        projected_outputs = []
        for i, output in enumerate(branch_outputs_list):
            projected = self.fusion.branch_projections[i](output)
            projected_outputs.append(projected)
        
        # Compute attention weights
        concatenated = torch.cat(projected_outputs, dim=-1)
        attention_weights = self.fusion.attention(concatenated)
        
        return attention_weights
