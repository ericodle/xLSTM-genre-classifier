"""
GAN-based data augmentation for dataset balancing.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gan.models import GanGenerator
from core.utils import setup_logging
from core.constants import DEFAULT_RANDOM_SEED

logger = logging.getLogger(__name__)


class GanAugmenter:
    """Augment datasets using trained GAN model."""

    def __init__(
        self,
        generator_path: str,
        noise_dim: int = 100,
        num_classes: int = 10,
        feature_dim: int = 13,
        hidden_dim: int = 256,
        num_layers: int = 3,
        device: str = "cuda",
        logger: logging.Logger = None,
    ):
        """
        Initialize GAN augmenter.

        Args:
            generator_path: Path to trained generator checkpoint
            noise_dim: Dimension of noise vector
            num_classes: Number of classes
            feature_dim: Feature dimension
            hidden_dim: Hidden dimension
            num_layers: Number of layers
            device: Device to use
            logger: Logger instance
        """
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.device = device
        self.logger = logger or logging.getLogger(__name__)

        # Load generator
        self.generator = GanGenerator(
            noise_dim=noise_dim,
            num_classes=num_classes,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(device)

        self.load_generator(generator_path)
        self.generator.eval()

    def load_generator(self, path: str):
        """Load generator from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.logger.info(f"Generator loaded from {path}")

    def generate_samples(
        self,
        num_samples: int,
        class_idx: int,
        random_seed: int = None,
        num_timesteps: int = 1292,
    ) -> np.ndarray:
        """
        Generate synthetic samples for a specific class.

        Args:
            num_samples: Number of samples to generate
            class_idx: Class index (0 to num_classes-1)
            random_seed: Random seed for reproducibility
            num_timesteps: Number of time steps to create (for 2D features)

        Returns:
            Generated features as numpy array
        """
        # Set random seed if provided
        if random_seed is not None:
            torch.manual_seed(random_seed)

        # Create class labels
        labels = torch.full((num_samples,), class_idx, dtype=torch.long).to(self.device)
        labels_onehot = F.one_hot(labels, self.num_classes).float()

        # Generate noise
        noise = torch.randn(num_samples, self.noise_dim).to(self.device)

        # Generate samples
        with torch.no_grad():
            generated_features = self.generator(noise, labels_onehot)

        # Convert to numpy and denormalize
        generated_features = generated_features.cpu().numpy()
        # Denormalize from [-1, 1] back to original scale
        generated_features = np.arctanh(generated_features.clip(-0.99, 0.99)) * 100.0

        # Reshape to 2D: (num_samples, num_timesteps, feature_dim)
        # Broadcast the 1D features across time dimension
        generated_features = np.tile(generated_features[:, np.newaxis, :], (1, num_timesteps, 1))

        return generated_features

    def balance_dataset(
        self,
        input_data_path: str,
        output_data_path: str,
        target_samples_per_class: int = None,
        balance_method: str = "equal",
    ) -> None:
        """
        Balance dataset using GAN-generated samples.

        Args:
            input_data_path: Path to input MFCC JSON file
            output_data_path: Path to output augmented MFCC JSON file
            target_samples_per_class: Target number of samples per class
            balance_method: Method for balancing ('equal', 'upsample', or specific number)
        """
        # Load original data
        self.logger.info(f"Loading data from {input_data_path}")
        with open(input_data_path, 'r') as f:
            data = json.load(f)

        # Get class distribution
        labels = data['labels']
        unique_labels = sorted(set(labels))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}

        # Calculate class distribution
        class_counts = {label: labels.count(label) for label in unique_labels}
        self.logger.info(f"Original class distribution: {class_counts}")

        # Determine target counts
        if balance_method == "equal":
            max_count = max(class_counts.values())
            target_counts = {label: max_count for label in unique_labels}
        elif balance_method == "upsample":
            target_counts = {label: int(count * 1.5) for label, count in class_counts.items()}
        elif isinstance(balance_method, int):
            target_counts = {label: balance_method for label in unique_labels}
        else:
            # Assume balance_method is target_samples_per_class
            target_counts = {label: target_samples_per_class for label in unique_labels}

        # Prepare features and labels
        existing_features = data['features']
        existing_labels = data['labels']

        # Generate synthetic samples for minority classes
        augmented_features = []
        augmented_labels = []

        for label in unique_labels:
            class_idx = label_to_idx[label]
            existing_count = class_counts[label]
            target_count = target_counts[label]
            needed_count = max(0, target_count - existing_count)

            self.logger.info(
                f"Class '{label}': existing={existing_count}, target={target_count}, "
                f"needed={needed_count}"
            )

            # Generate synthetic samples if needed
            if needed_count > 0:
                self.logger.info(f"Generating {needed_count} synthetic samples for class '{label}'")
                synthetic_samples = self.generate_samples(
                    num_samples=needed_count,
                    class_idx=class_idx,
                    random_seed=DEFAULT_RANDOM_SEED,
                )
                synthetic_samples = synthetic_samples.tolist()

                augmented_features.extend(synthetic_samples)
                augmented_labels.extend([label] * needed_count)

        # Combine original and synthetic data
        all_features = existing_features + augmented_features
        all_labels = existing_labels + augmented_labels

        # Shuffle data
        indices = np.random.permutation(len(all_features))
        shuffled_features = [all_features[i] for i in indices]
        shuffled_labels = [all_labels[i] for i in indices]

        # Update class distribution
        final_class_counts = {label: shuffled_labels.count(label) for label in unique_labels}
        self.logger.info(f"Final class distribution: {final_class_counts}")

        # Save augmented data
        output_data = {
            'features': shuffled_features,
            'labels': shuffled_labels,
            'mapping': data.get('mapping', unique_labels),
            'augmented': True,
            'original_samples': len(existing_features),
            'synthetic_samples': len(augmented_features),
            'augmentation_ratio': len(augmented_features) / len(existing_features),
        }

        os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
        with open(output_data_path, 'w') as f:
            json.dump(output_data, f)

        self.logger.info(f"Augmented data saved to {output_data_path}")
        self.logger.info(
            f"Original: {len(existing_features)}, Synthetic: {len(augmented_features)}, "
            f"Total: {len(all_features)}"
        )

    def generate_single_class(
        self,
        class_idx: int,
        num_samples: int,
        output_path: str,
        random_seed: int = None,
    ) -> None:
        """
        Generate samples for a single class and save to file.

        Args:
            class_idx: Class index
            num_samples: Number of samples to generate
            output_path: Output file path
            random_seed: Random seed
        """
        generated_features = self.generate_samples(
            num_samples=num_samples,
            class_idx=class_idx,
            random_seed=random_seed,
        )

        # Save to JSON
        data = {
            'features': generated_features.tolist(),
            'class_idx': class_idx,
            'num_samples': num_samples,
        }

        with open(output_path, 'w') as f:
            json.dump(data, f)

        self.logger.info(f"Generated {num_samples} samples for class {class_idx}")
        self.logger.info(f"Saved to {output_path}")


def main():
    """Main augmentation function."""
    parser = argparse.ArgumentParser(description="Augment dataset using trained GAN")
    parser.add_argument(
        "--generator",
        type=str,
        required=True,
        help="Path to trained generator checkpoint",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input MFCC JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output augmented MFCC JSON file",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="equal",
        choices=["equal", "upsample", "custom"],
        help="Balancing method",
    )
    parser.add_argument(
        "--target-samples",
        type=int,
        help="Target samples per class (for equal method)",
    )
    parser.add_argument(
        "--noise-dim",
        type=int,
        default=100,
        help="Noise dimension",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of layers",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )

    args = parser.parse_args()

    # Setup
    setup_logging()
    logger.info("Starting GAN-based augmentation")

    # Load data to determine parameters
    with open(args.input, 'r') as f:
        data = json.load(f)

    unique_labels = sorted(set(data['labels']))
    num_classes = len(unique_labels)
    
    # Calculate feature_dim correctly - take mean of 2D features to get 1D
    import numpy as np
    if data['features']:
        sample_feature = np.array(data['features'][0])
        if len(sample_feature.shape) == 2:
            # 2D MFCC: take mean across time dimension to get 13 features
            feature_dim = sample_feature.shape[1]  # Second dimension
        else:
            feature_dim = len(sample_feature)
    else:
        feature_dim = 13

    # Create augmenter
    augmenter = GanAugmenter(
        generator_path=args.generator,
        noise_dim=args.noise_dim,
        num_classes=num_classes,
        feature_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        device=args.device,
    )

    # Augment dataset
    target_samples = args.target_samples if args.method == "custom" else None
    augmenter.balance_dataset(
        input_data_path=args.input,
        output_data_path=args.output,
        target_samples_per_class=target_samples,
        balance_method=args.method if args.method != "custom" else target_samples,
    )

    logger.info("Augmentation completed!")


if __name__ == "__main__":
    main()

