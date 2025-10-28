"""
Advanced loss functions for music genre classification.
Includes label smoothing to handle genre overlap and reduce overfitting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropyLoss(nn.Module):
    """
    Label smoothing cross-entropy loss.

    Instead of hard one-hot labels [1, 0, 0, ..., 0], uses soft labels like:
    [0.9, 0.05, 0.0, ..., 0.05]

    Benefits:
    - Reduces overfitting by preventing overconfident predictions
    - Handles genre overlap/ambiguity naturally
    - Improves generalization
    - Simple regularization technique

    Args:
        smoothing: Label smoothing factor (0.0 = hard labels, 1.0 = uniform)
        num_classes: Number of classes
    """

    def __init__(self, smoothing: float = 0.1, num_classes: int = 10):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.confidence = 1.0 - smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted logits (batch_size, num_classes)
            target: Target class indices (batch_size,)

        Returns:
            Smoothed loss
        """
        # Convert predictions to log probabilities
        log_probs = F.log_softmax(pred, dim=1)

        # Create smoothed labels
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            # Fill all classes with smoothing / (num_classes - 1)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            # Set correct class to confidence value
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        # Compute KL divergence (or equivalently, cross-entropy with soft labels)
        loss = -torch.sum(true_dist * log_probs, dim=1)

        return loss.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Focuses learning on hard examples.
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
