"""
Training script for GAN model.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gan.models import GanGenerator, GanDiscriminator
from core.utils import setup_logging
from core.constants import DEFAULT_DEVICE, DEFAULT_RANDOM_SEED

logger = logging.getLogger(__name__)


class MFCCDataset(Dataset):
    """Dataset for MFCC features with class labels."""

    def __init__(self, data_path: str, flatten_features: bool = True):
        """
        Initialize dataset.

        Args:
            data_path: Path to MFCC JSON file
            flatten_features: Whether to flatten MFCC features to 1D
        """
        self.flatten_features = flatten_features
        self.features, self.labels, self.class_to_idx = self._load_data(data_path)

    def _load_data(self, data_path: str) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
        """Load MFCC data from JSON file."""
        with open(data_path, 'r') as f:
            data = json.load(f)

        features_list = []
        labels_list = []

        # Get unique genres
        unique_labels = sorted(set(data['labels']))
        class_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

        for feature, label in zip(data['features'], data['labels']):
            feature_array = np.array(feature)
            
            # Flatten if necessary (convert 2D MFCC to 1D)
            if self.flatten_features and len(feature_array.shape) > 1:
                # Average or take first timestep
                feature_array = feature_array.mean(axis=0)
            
            # Normalize features to [-1, 1] range
            feature_array = np.tanh(feature_array / 100.0)
            
            features_list.append(feature_array)
            labels_list.append(class_to_idx[label])

        features = torch.FloatTensor(np.array(features_list))
        labels = torch.LongTensor(labels_list)

        return features, labels, class_to_idx

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class GanTrainer:
    """Trainer for WGAN-GP model."""

    def __init__(
        self,
        generator: GanGenerator,
        discriminator: GanDiscriminator,
        noise_dim: int = 100,
        num_classes: int = 10,
        lambda_gp: float = 10.0,
        n_critic: int = 5,
        device: str = "cuda",
        logger: logging.Logger = None,
    ):
        """
        Initialize GAN trainer.

        Args:
            generator: Generator model
            discriminator: Discriminator model
            noise_dim: Dimension of noise vector
            num_classes: Number of classes
            lambda_gp: Gradient penalty coefficient
            n_critic: Number of critic iterations per generator iteration
            device: Device to train on
            logger: Logger instance
        """
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.lambda_gp = lambda_gp
        self.n_critic = n_critic
        self.device = device
        self.logger = logger or logging.getLogger(__name__)

        # Optimizers (using RMSprop for stability)
        self.g_optimizer = optim.RMSprop(
            self.generator.parameters(),
            lr=0.00005
        )
        self.d_optimizer = optim.RMSprop(
            self.discriminator.parameters(),
            lr=0.00005
        )

        # Training history
        self.history = {
            'g_loss': [],
            'd_loss': [],
            'real_scores': [],
            'fake_scores': [],
        }

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        g_losses = []
        d_losses = []
        real_scores = []
        fake_scores = []

        for batch_idx, (features, labels) in enumerate(dataloader):
            batch_size = features.size(0)
            features = features.to(self.device)
            labels = labels.to(self.device)

            # Create one-hot encoded labels
            labels_onehot = F.one_hot(labels, self.num_classes).float()

            # Train discriminator (critic)
            d_loss = 0.0
            for _ in range(self.n_critic):
                # Train with real data
                real_scores_batch = self.discriminator(features, labels_onehot)

                # Train with fake data
                noise = torch.randn(batch_size, self.noise_dim).to(self.device)
                fake_features = self.generator(noise, labels_onehot)
                fake_scores_batch = self.discriminator(fake_features.detach(), labels_onehot)

                # Get gradient penalty
                _, _, gp = self.discriminator.forward_with_gradient_penalty(
                    features, fake_features, labels_onehot, labels_onehot, self.device
                )

                # Discriminator loss
                d_loss_batch = (
                    real_scores_batch.mean() - fake_scores_batch.mean() + self.lambda_gp * gp
                )

                # Update discriminator
                self.d_optimizer.zero_grad()
                d_loss_batch.backward()
                self.d_optimizer.step()

                d_loss += d_loss_batch.item()

            d_loss /= self.n_critic

            # Train generator
            noise = torch.randn(batch_size, self.noise_dim).to(self.device)
            fake_features = self.generator(noise, labels_onehot)
            fake_scores_g = self.discriminator(fake_features, labels_onehot)

            # Generator loss (maximize fake score, i.e., minimize negative)
            g_loss = -fake_scores_g.mean()

            self.g_optimizer.zero_grad()
            g_loss.backward()
            self.g_optimizer.step()

            # Store losses
            g_losses.append(g_loss.item())
            d_losses.append(d_loss)
            real_scores.append(real_scores_batch.mean().item())
            fake_scores.append(fake_scores_batch.mean().item())

        return {
            'g_loss': np.mean(g_losses),
            'd_loss': np.mean(d_losses),
            'real_score': np.mean(real_scores),
            'fake_score': np.mean(fake_scores),
        }

    def train(
        self,
        train_loader: DataLoader,
        epochs: int = 100,
        save_interval: int = 10,
        output_dir: str = "outputs/gan_training",
    ):
        """Train GAN model."""
        os.makedirs(output_dir, exist_ok=True)

        self.logger.info(f"Starting GAN training for {epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters())}")
        self.logger.info(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters())}")

        best_loss = float('inf')
        for epoch in range(epochs):
            metrics = self.train_epoch(train_loader)

            # Log metrics
            self.history['g_loss'].append(metrics['g_loss'])
            self.history['d_loss'].append(metrics['d_loss'])
            self.history['real_scores'].append(metrics['real_score'])
            self.history['fake_scores'].append(metrics['fake_score'])

            if (epoch + 1) % 10 == 0:
                self.logger.info(
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"G_loss: {metrics['g_loss']:.4f} "
                    f"D_loss: {metrics['d_loss']:.4f} "
                    f"Real: {metrics['real_score']:.4f} "
                    f"Fake: {metrics['fake_score']:.4f}"
                )

            # Save model
            if (epoch + 1) % save_interval == 0 or metrics['g_loss'] < best_loss:
                best_loss = min(best_loss, metrics['g_loss'])
                self.save_checkpoint(
                    epoch + 1,
                    os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pth"),
                )

        # Save final model
        self.save_checkpoint(epochs, os.path.join(output_dir, "final_model.pth"))
        self.save_history(os.path.join(output_dir, "training_history.json"))

        self.logger.info("Training completed!")

    def save_checkpoint(self, epoch: int, path: str):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'history': self.history,
        }, path)
        self.logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        self.logger.info(f"Checkpoint loaded from {path}")

    def save_history(self, path: str):
        """Save training history to JSON."""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        self.logger.info(f"History saved to {path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train GAN for data augmentation")
    parser.add_argument("--data", type=str, required=True, help="Path to MFCC JSON file")
    parser.add_argument("--output", type=str, default="outputs/gan_training", help="Output directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--noise-dim", type=int, default=100, help="Noise vector dimension")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of hidden layers")
    parser.add_argument("--n-critic", type=int, default=5, help="Critic iterations per generator")
    parser.add_argument("--lambda-gp", type=float, default=10.0, help="Gradient penalty coefficient")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed")

    args = parser.parse_args()

    # Setup
    setup_logging()
    logger.info("Starting GAN training")
    logger.info(f"Arguments: {args}")

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    logger.info(f"Loading data from {args.data}")
    dataset = MFCCDataset(args.data)
    
    # Determine number of classes
    num_classes = len(dataset.class_to_idx)
    feature_dim = dataset.features.shape[1]
    
    logger.info(f"Dataset: {len(dataset)} samples, {num_classes} classes, {feature_dim} features")

    # Create data loader
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )

    # Create models
    generator = GanGenerator(
        noise_dim=args.noise_dim,
        num_classes=num_classes,
        feature_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )
    
    discriminator = GanDiscriminator(
        feature_dim=feature_dim,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )

    # Create trainer
    trainer = GanTrainer(
        generator=generator,
        discriminator=discriminator,
        noise_dim=args.noise_dim,
        num_classes=num_classes,
        lambda_gp=args.lambda_gp,
        n_critic=args.n_critic,
        device=args.device,
    )

    # Train
    trainer.train(
        train_loader=train_loader,
        epochs=args.epochs,
        save_interval=10,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()

