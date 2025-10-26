"""
Multimodal music genre classification module.
Implements specialized branches for different types of audio features.
"""

from .feature_extractor import MultimodalFeatureExtractor, MultimodalFeatures
from .model_branches import SpectralCNNBranch, TemporalRNNBranch, StatisticalFCBranch, FeatureProcessor
from .multimodal_model import MultimodalModel, AttentionFusion, ConcatFusion
from .train_multimodal import MultimodalTrainer, MultimodalDataset

__all__ = [
    'MultimodalFeatureExtractor',
    'MultimodalFeatures',
    'SpectralCNNBranch',
    'TemporalRNNBranch', 
    'StatisticalFCBranch',
    'FeatureProcessor',
    'MultimodalModel',
    'AttentionFusion',
    'ConcatFusion',
    'MultimodalTrainer',
    'MultimodalDataset'
]
