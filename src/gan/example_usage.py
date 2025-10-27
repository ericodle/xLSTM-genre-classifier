"""
Example usage of GAN module for data augmentation.
"""

import os
import sys
import logging

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gan.train_gan import GanTrainer, MFCCDataset
from gan.augment import GanAugmenter
from torch.utils.data import DataLoader
from core.utils import setup_logging

logger = logging.getLogger(__name__)


def train_gan_example():
    """Example: Train a GAN on GTZAN dataset."""
    print("=" * 60)
    print("Example 1: Training a GAN on GTZAN dataset")
    print("=" * 60)
    
    # Configuration
    data_path = "mfccs/gtzan_13.json"
    output_dir = "outputs/gan_gtzan"
    epochs = 100
    batch_size = 64
    
    print(f"\nTraining GAN on {data_path}")
    print(f"Output directory: {output_dir}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    
    # Load dataset
    dataset = MFCCDataset(data_path)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    
    print(f"\nDataset loaded: {len(dataset)} samples")
    print(f"Feature dimension: {dataset.features.shape[1]}")
    print(f"Number of classes: {len(dataset.class_to_idx)}")
    print(f"Classes: {list(dataset.class_to_idx.keys())}")
    
    # Note: Actual training would happen here
    print("\nTo train the GAN, run:")
    print(f"  python src/gan/train_gan.py --data {data_path} --output {output_dir} --epochs {epochs}")


def augment_example():
    """Example: Using trained GAN to augment imbalanced dataset."""
    print("\n" + "=" * 60)
    print("Example 2: Using GAN to augment imbalanced dataset")
    print("=" * 60)
    
    # Configuration
    generator_path = "outputs/gan_gtzan/final_model.pth"
    input_data = "mfccs/gtzan_13.json"
    output_data = "mfccs/gtzan_13_augmented.json"
    
    print(f"\nGenerator: {generator_path}")
    print(f"Input data: {input_data}")
    print(f"Output data: {output_data}")
    
    # Note: Actual augmentation would happen here
    print("\nTo augment the dataset, run:")
    print(f"  python src/gan/augment.py \\")
    print(f"    --generator {generator_path} \\")
    print(f"    --input {input_data} \\")
    print(f"    --output {output_data} \\")
    print(f"    --method equal")


def complete_workflow_example():
    """Example: Complete workflow from training to augmentation."""
    print("\n" + "=" * 60)
    print("Example 3: Complete GAN Workflow")
    print("=" * 60)
    
    workflow = """
    1. Train GAN on existing dataset:
       python src/gan/train_gan.py \\
         --data mfccs/gtzan_13.json \\
         --output outputs/gan_gtzan \\
         --epochs 100 \\
         --batch-size 64

    2. Augment imbalanced dataset:
       python src/gan/augment.py \\
         --generator outputs/gan_gtzan/final_model.pth \\
         --input mfccs/gtzan_13.json \\
         --output mfccs/gtzan_13_augmented.json \\
         --method equal

    3. Train classifier on augmented data:
       python src/training/train_model.py \\
         --data mfccs/gtzan_13_augmented.json \\
         --model LSTM \\
         --output outputs/classifier_gan_augmented

    Benefits:
    - Balances imbalanced datasets
    - Improves classifier performance on minority classes
    - Adds variety to training data without collecting new samples
    """
    
    print(workflow)


def main():
    """Run all examples."""
    setup_logging()
    
    print("\nGAN Module - Example Usage")
    print("=" * 60)
    
    train_gan_example()
    augment_example()
    complete_workflow_example()
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()

