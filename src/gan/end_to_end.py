#!/usr/bin/env python3
"""
End-to-end GAN training and augmentation pipeline.
This script will:
1. Train a GAN on the dataset
2. Augment the dataset
3. Train a classifier on the augmented dataset
"""

import os
import sys
import logging
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.utils import setup_logging

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*70}")
    print(f"Step: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: {description} failed!")
        print(result.stderr)
        return False
    
    print(result.stdout)
    return True

def main():
    """Run end-to-end pipeline."""
    print("="*70)
    print("GAN-Based Dataset Augmentation Pipeline")
    print("="*70)
    
    # Configuration
    base_data = "mfccs/gtzan_13.json"
    gan_output = "outputs/gan_gtzan_v2"
    augmented_data = "mfccs/gtzan_13_augmented.json"
    classifier_output = "outputs/classifier_augmented"
    
    epochs = 50
    batch_size = 64
    
    # Step 1: Train GAN
    print(f"\nStep 1: Training GAN on {base_data}")
    cmd = [
        "python", "src/gan/train_gan.py",
        "--data", base_data,
        "--output", gan_output,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
    ]
    
    if not run_command(cmd, "Train GAN"):
        return
    
    # Step 2: Augment dataset
    print(f"\nStep 2: Augmenting dataset")
    cmd = [
        "python", "src/gan/augment.py",
        "--generator", os.path.join(gan_output, "final_model.pth"),
        "--input", base_data,
        "--output", augmented_data,
        "--method", "equal",
        "--noise-dim", "100",
        "--hidden-dim", "256",
        "--num-layers", "3",
    ]
    
    if not run_command(cmd, "Augment dataset"):
        return
    
    # Step 3: Train classifier
    print(f"\nStep 3: Training classifier on augmented data")
    cmd = [
        "python", "src/training/train_model.py",
        "--data", augmented_data,
        "--model", "LSTM",
        "--output", classifier_output,
    ]
    
    if not run_command(cmd, "Train classifier"):
        return
    
    print(f"\n{'='*70}")
    print("Pipeline completed successfully!")
    print(f"{'='*70}")
    print(f"\nResults:")
    print(f"  - GAN training: {gan_output}")
    print(f"  - Augmented dataset: {augmented_data}")
    print(f"  - Classifier: {classifier_output}")
    print(f"\nNext steps:")
    print(f"  - Evaluate: python src/training/evaluate_model.py --model {classifier_output}/final_model.pth")
    print(f"  - Analyze: python src/analysis/analyze_results.py --input-dir {classifier_output}")

if __name__ == "__main__":
    setup_logging()
    main()

