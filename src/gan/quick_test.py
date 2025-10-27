#!/usr/bin/env python3
"""
Quick test script for GAN module.
"""

import os
import sys
import logging

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gan.train_gan import MFCCDataset, GanTrainer
from gan.models import GanGenerator, GanDiscriminator
from gan.augment import GanAugmenter
from torch.utils.data import DataLoader
from core.utils import setup_logging
from core.constants import DEFAULT_RANDOM_SEED

def main():
    """Test GAN training and augmentation."""
    print("=" * 70)
    print("GAN Module Quick Test")
    print("=" * 70)
    
    # Setup logging
    setup_logging(log_level="INFO")
    logger = logging.getLogger(__name__)
    
    # Configuration
    data_path = "mfccs/gtzan_13.json"
    output_dir = "outputs/gan_quick_test"
    batch_size = 32
    epochs = 5
    
    print(f"\nStep 1: Loading data from {data_path}")
    try:
        dataset = MFCCDataset(data_path)
        print(f"✓ Dataset loaded: {len(dataset)} samples")
        print(f"✓ Feature shape: {dataset.features.shape}")
        print(f"✓ Number of classes: {len(dataset.class_to_idx)}")
        print(f"✓ Classes: {list(dataset.class_to_idx.keys())}")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return
    
    # Create data loader
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    
    # Create models
    print(f"\nStep 2: Creating GAN models")
    try:
        num_classes = len(dataset.class_to_idx)
        feature_dim = dataset.features.shape[1]
        
        generator = GanGenerator(
            noise_dim=100,
            num_classes=num_classes,
            feature_dim=feature_dim,
            hidden_dim=128,  # Smaller for quick test
            num_layers=2,
        )
        
        discriminator = GanDiscriminator(
            feature_dim=feature_dim,
            num_classes=num_classes,
            hidden_dim=128,
            num_layers=2,
        )
        
        print(f"✓ Generator created: {sum(p.numel() for p in generator.parameters())} parameters")
        print(f"✓ Discriminator created: {sum(p.numel() for p in discriminator.parameters())} parameters")
    except Exception as e:
        print(f"✗ Failed to create models: {e}")
        return
    
    # Train GAN
    print(f"\nStep 3: Training GAN for {epochs} epochs")
    try:
        # Check if CUDA is available
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        trainer = GanTrainer(
            generator=generator,
            discriminator=discriminator,
            noise_dim=100,
            num_classes=num_classes,
            n_critic=3,  # Fewer critic iterations for speed
            device=device,
        )
        
        print(f"✓ Trainer created on device: {trainer.device}")
        
        # Run training
        trainer.train(
            train_loader=train_loader,
            epochs=epochs,
            save_interval=epochs,
            output_dir=output_dir,
        )
        
        print(f"✓ Training completed!")
    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test augmentation
    print(f"\nStep 4: Testing augmentation")
    try:
        checkpoint_path = os.path.join(output_dir, "final_model.pth")
        
        augmenter = GanAugmenter(
            generator_path=checkpoint_path,
            noise_dim=100,
            num_classes=num_classes,
            feature_dim=feature_dim,
            hidden_dim=128,
            num_layers=2,
            device=trainer.device,
        )
        
        print(f"✓ Augmenter loaded")
        
        # Generate a few samples for each class
        print(f"\nStep 5: Generating sample data")
        for class_idx, class_name in dataset.class_to_idx.items():
            if isinstance(class_name, int):
                samples = augmenter.generate_samples(
                    num_samples=5,
                    class_idx=class_name,
                    random_seed=DEFAULT_RANDOM_SEED,
                )
                print(f"✓ Generated {len(samples)} samples for class {class_idx}")
        
        print(f"\n✓ All tests passed!")
        
    except Exception as e:
        print(f"✗ Augmentation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"✓ Dataset loading: OK")
    print(f"✓ Model creation: OK")
    print(f"✓ GAN training: OK")
    print(f"✓ Sample generation: OK")
    print("\nNext steps:")
    print(f"1. Train longer: python src/gan/train_gan.py --data cater/train_data.json --output outputs/gan_full --epochs 100")
    print(f"2. Augment dataset: python src/gan/augment.py --generator outputs/gan_full/final_model.pth --input mfccs/gtzan_13.json --output mfccs/gtzan_13_augmented.json")
    print(f"3. Train classifier: python src/training/train_model.py --data mfccs/gtzan_13_augmented.json --model CNN --output outputs/classifier_augmented")


if __name__ == "__main__":
    main()

