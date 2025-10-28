"""
Vision Transformer (ViT) classifier for MFCC inputs.

We use timm's Vision Transformer and adapt it to work with audio spectrograms (MFCC features).
"""

import torch
import torch.nn as nn

# Check if timm is available
try:
    import timm

    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm library not found. Install with: pip install timm")

import os

# Add src directory to path for imports
import sys

from .base import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.constants import DEFAULT_DROPOUT, DEFAULT_NUM_CLASSES


class ViTClassifier(BaseModel):
    """
    Vision Transformer classifier for MFCC audio features.

    This model treats MFCC spectrograms as images and uses a Vision Transformer
    to extract features and classify audio genres.
    """

    def __init__(
        self,
        num_classes: int = DEFAULT_NUM_CLASSES,
        pretrained: bool = True,
        dropout: float = DEFAULT_DROPOUT,
        num_mfcc_features: int = 13,
        image_size: int = 224,
        patch_size: int = 16,
        dim: int = 768,
        depth: int = 12,
        heads: int = 12,
        mlp_dim: int = 3072,
    ):
        """
        Initialize Vision Transformer classifier.

        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained ImageNet weights
            dropout: Dropout probability
            num_mfcc_features: Number of MFCC coefficients (height of spectrogram)
            image_size: Size of input image (will be padded/interpolated if needed)
            patch_size: Size of patches to split image into
            dim: Dimension of transformer embeddings
            depth: Number of transformer layers
            heads: Number of attention heads
            mlp_dim: Dimension of MLP in transformer blocks
        """
        super().__init__(model_name="ViT")

        if not TIMM_AVAILABLE:
            raise ImportError("timm library is required for ViT. Install with: pip install timm")

        self.num_classes = num_classes
        self.dropout = dropout
        self.num_mfcc_features = num_mfcc_features
        self.image_size = image_size
        self.patch_size = patch_size

        # Load ViT base model from timm
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=0,  # Remove default classifier
            img_size=image_size,
        )

        # Store original embedding for patching
        self.embed_dim = self.vit.embed_dim

        # Create a projection layer to adapt 1-channel MFCC to 3-channel RGB
        # We'll replicate the single channel 3 times to match ImageNet pretrained weights
        self.input_proj = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0, bias=False)

        # Custom classifier head
        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(mlp_dim, num_classes),
        )

        # Store config
        self.model_config = {
            "num_classes": num_classes,
            "pretrained": pretrained,
            "dropout": dropout,
            "image_size": image_size,
            "patch_size": patch_size,
            "dim": dim,
            "depth": depth,
            "heads": heads,
            "mlp_dim": mlp_dim,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Vision Transformer.

        Args:
            x: Input tensor of shape (batch, time, features) or (batch, 1, time, features)

        Returns:
            Log probabilities for each class
        """
        # Handle different input shapes
        if x.dim() == 3:
            # (batch, time, features) -> (batch, 1, time, features)
            x = x.unsqueeze(1)
        elif x.dim() == 4 and x.shape[1] != 1:
            # If given (batch, H, W) format, add channel
            x = x.unsqueeze(1)

        # Now x should be (batch, channels, time, features)
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor after preprocessing, got {x.dim()}D")

        batch_size = x.shape[0]
        time_steps = x.shape[2]
        features = x.shape[3]

        # Resize/interpolate to match ViT input size (image_size x image_size)
        # MFCC input is typically (1, time_steps, num_mfcc_features)
        # We need to resize to (image_size, image_size)
        if time_steps != self.image_size or features != self.image_size:
            x = torch.nn.functional.interpolate(
                x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
            )

        # Project single channel to 3 channels (RGB) to use pretrained weights
        x = self.input_proj(x)

        # Pass through ViT backbone (excluding the original classifier)
        # timm's ViT returns features from all patches including CLS token
        features = self.vit.forward_features(x)

        # Extract CLS token (first token) from the sequence
        if len(features.shape) == 3:
            # features is (batch, seq_len, dim)
            cls_features = features[:, 0]  # Take first token (CLS token)
        else:
            cls_features = features

        # Apply custom classifier
        out = self.head(cls_features)

        return torch.log_softmax(out, dim=1)
