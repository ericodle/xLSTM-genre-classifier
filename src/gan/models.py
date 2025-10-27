"""
GAN models for audio feature generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class GanGenerator(nn.Module):
    """
    Generator for WGAN-GP model.
    Generates MFCC features from noise vectors.
    """

    def __init__(
        self,
        noise_dim: int = 100,
        num_classes: int = 10,
        feature_dim: int = 13,  # MFCC coefficients
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        """
        Initialize Generator.

        Args:
            noise_dim: Dimension of noise vector
            num_classes: Number of genre classes
            hidden_dim: Hidden layer dimension
            feature_dim: Output feature dimension (MFCC coefficients)
            num_layers: Number of hidden layers
            dropout: Dropout probability
        """
        super(GanGenerator, self).__init__()

        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # Concatenate noise and class embedding
        self.input_dim = noise_dim + num_classes

        # Build generator layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(self.input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for i in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_dim, feature_dim))
        layers.append(nn.Tanh())  # Normalize output to [-1, 1]

        self.model = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, z: torch.Tensor, class_labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            z: Noise tensor of shape (batch_size, noise_dim)
            class_labels: One-hot encoded class labels of shape (batch_size, num_classes)

        Returns:
            Generated features of shape (batch_size, feature_dim)
        """
        # Concatenate noise and class label
        x = torch.cat([z, class_labels], dim=1)
        
        # Generate features
        output = self.model(x)
        
        return output


class GanDiscriminator(nn.Module):
    """
    Discriminator for WGAN-GP model.
    Distinguishes real from fake MFCC features.
    """

    def __init__(
        self,
        feature_dim: int = 13,  # MFCC coefficients
        num_classes: int = 10,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        """
        Initialize Discriminator.

        Args:
            feature_dim: Input feature dimension (MFCC coefficients)
            num_classes: Number of genre classes
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            dropout: Dropout probability
        """
        super(GanDiscriminator, self).__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # Input dimension includes features and class embedding
        self.input_dim = feature_dim + num_classes

        # Build discriminator layers
        layers = []

        # Input layer
        layers.append(nn.Linear(self.input_dim, hidden_dim))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for i in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout))

        # Output layer (real/fake score)
        layers.append(nn.Linear(hidden_dim, 1))
        # No sigmoid for WGAN-GP - we use raw scores

        self.model = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, features: torch.Tensor, class_labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: Feature tensor of shape (batch_size, feature_dim)
            class_labels: One-hot encoded class labels of shape (batch_size, num_classes)

        Returns:
            Discriminator score of shape (batch_size, 1)
        """
        # Concatenate features and class labels
        x = torch.cat([features, class_labels], dim=1)

        # Get discriminator score
        output = self.model(x)

        return output

    def forward_with_gradient_penalty(
        self, 
        real_features: torch.Tensor,
        fake_features: torch.Tensor,
        real_labels: torch.Tensor,
        fake_labels: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with gradient penalty for WGAN-GP.

        Args:
            real_features: Real features
            fake_features: Generated features
            real_labels: Real class labels
            fake_labels: Fake class labels
            device: Device to compute on

        Returns:
            Tuple of (real scores, fake scores, gradient penalty)
        """
        batch_size = real_features.size(0)

        # Get real and fake scores
        real_scores = self.forward(real_features, real_labels)
        fake_scores = self.forward(fake_features, fake_labels)

        # Create interpolated samples - must be on same device as features
        alpha = torch.rand(batch_size, 1, device=real_features.device)
        interpolated_features = (alpha * real_features + (1 - alpha) * fake_features)
        interpolated_labels = (alpha * real_labels + (1 - alpha) * fake_labels)
        
        # Set requires_grad on interpolated features BEFORE forward pass
        interpolated_features.requires_grad_(True)

        # Get discriminator score for interpolated samples
        interpolated_scores = self.forward(interpolated_features, interpolated_labels)

        # Compute gradient penalty with proper device handling
        grad_outputs = torch.ones_like(interpolated_scores, device=interpolated_scores.device)
        gradients = torch.autograd.grad(
            outputs=interpolated_scores,
            inputs=interpolated_features,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return real_scores, fake_scores, gradient_penalty


