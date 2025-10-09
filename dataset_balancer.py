#!/usr/bin/env python3
"""
Dataset Balancer for GenreDiscern

This script balances imbalanced datasets by augmenting underrepresented classes
using various data augmentation techniques specific to MFCC features.
"""

import json
import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, List, Tuple, Any
import argparse
import os
from pathlib import Path


class MFCCAugmenter:
    """Augmentation techniques for MFCC features."""
    
    @staticmethod
    def add_noise(mfcc: np.ndarray, noise_factor: float = 0.01) -> np.ndarray:
        """Add Gaussian noise to MFCC features."""
        noise = np.random.normal(0, noise_factor, mfcc.shape)
        return mfcc + noise
    
    @staticmethod
    def time_shift(mfcc: np.ndarray, shift_factor: float = 0.1) -> np.ndarray:
        """Apply time shifting by rolling the sequence."""
        seq_len = mfcc.shape[0]
        shift_amount = int(seq_len * shift_factor * np.random.uniform(-1, 1))
        return np.roll(mfcc, shift_amount, axis=0)
    
    @staticmethod
    def pitch_shift(mfcc: np.ndarray, shift_factor: float = 0.1) -> np.ndarray:
        """Apply pitch shifting by scaling MFCC coefficients."""
        # Scale MFCC coefficients to simulate pitch change
        scale_factor = 1 + np.random.uniform(-shift_factor, shift_factor)
        return mfcc * scale_factor
    
    @staticmethod
    def time_stretch(mfcc: np.ndarray, stretch_factor: float = 0.1) -> np.ndarray:
        """Apply time stretching by interpolating the sequence."""
        seq_len = mfcc.shape[0]
        stretch_amount = 1 + np.random.uniform(-stretch_factor, stretch_factor)
        new_length = int(seq_len * stretch_amount)
        
        if new_length == seq_len:
            return mfcc
        
        # Interpolate to new length
        indices = np.linspace(0, seq_len - 1, new_length)
        stretched = np.zeros((new_length, mfcc.shape[1]))
        
        for i in range(mfcc.shape[1]):
            stretched[:, i] = np.interp(indices, np.arange(seq_len), mfcc[:, i])
        
        return stretched
    
    @staticmethod
    def frequency_mask(mfcc: np.ndarray, mask_factor: float = 0.1) -> np.ndarray:
        """Apply frequency masking by zeroing out some coefficients."""
        masked = mfcc.copy()
        mask_size = int(mfcc.shape[1] * mask_factor)
        if mask_size > 0:
            start_idx = np.random.randint(0, mfcc.shape[1] - mask_size + 1)
            masked[:, start_idx:start_idx + mask_size] = 0
        return masked
    
    @staticmethod
    def mixup(mfcc1: np.ndarray, mfcc2: np.ndarray, alpha: float = 0.2) -> np.ndarray:
        """Mix two MFCC samples using mixup technique."""
        # Ensure same length
        min_len = min(mfcc1.shape[0], mfcc2.shape[0])
        mfcc1 = mfcc1[:min_len]
        mfcc2 = mfcc2[:min_len]
        
        lam = np.random.beta(alpha, alpha)
        return lam * mfcc1 + (1 - lam) * mfcc2


class DatasetBalancer:
    """Balances datasets by augmenting underrepresented classes."""
    
    def __init__(self, augmentation_methods: List[str] = None):
        """
        Initialize the dataset balancer.
        
        Args:
            augmentation_methods: List of augmentation methods to use
        """
        self.augmentation_methods = augmentation_methods or [
            'add_noise', 'time_shift', 'pitch_shift', 'time_stretch', 'frequency_mask'
        ]
        self.augmenter = MFCCAugmenter()
    
    def load_dataset(self, filepath: str) -> Tuple[List[np.ndarray], List[str]]:
        """Load dataset from JSON file."""
        print(f"Loading dataset from {filepath}...")
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Handle different data formats
        if isinstance(data, dict) and 'features' in data and 'labels' in data:
            # Format: {"features": [...], "labels": [...]}
            features = [np.array(f) for f in data['features']]
            labels = data['labels']
        elif isinstance(data, list):
            # Format: [{"mfcc": [...], "genre": "..."}, ...]
            features = []
            labels = []
            for item in data:
                if 'mfcc' in item and 'genre' in item:
                    features.append(np.array(item['mfcc']))
                    labels.append(item['genre'])
        else:
            raise ValueError(f"Unsupported data format in {filepath}")
        
        print(f"Loaded {len(features)} samples with {len(set(labels))} classes")
        return features, labels
    
    def analyze_class_distribution(self, labels: List[str]) -> Dict[str, int]:
        """Analyze the distribution of classes."""
        class_counts = Counter(labels)
        total_samples = len(labels)
        
        print("\nClass Distribution:")
        print("-" * 40)
        for genre, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_samples) * 100
            print(f"{str(genre):20s}: {count:4d} samples ({percentage:5.1f}%)")
        
        return dict(class_counts)
    
    def calculate_target_samples(self, class_counts: Dict[str, int], 
                               target_ratio: float = 0.1) -> Dict[str, int]:
        """
        Calculate how many samples each class needs to reach target ratio.
        
        Args:
            class_counts: Current class distribution
            target_ratio: Target ratio for minority classes (0.1 = 10% of max class)
        
        Returns:
            Dictionary with target sample counts for each class
        """
        max_samples = max(class_counts.values())
        target_samples = {}
        
        # More conservative approach: limit maximum augmentation
        max_augmentation_factor = 2.0  # Don't more than double any class
        
        for genre, count in class_counts.items():
            if count < max_samples * target_ratio:
                # Calculate target but cap it
                target = int(max_samples * target_ratio)
                max_allowed = int(count * max_augmentation_factor)
                target_samples[genre] = min(target, max_allowed)
            else:
                target_samples[genre] = count
        
        return target_samples
    
    def augment_sample(self, mfcc: np.ndarray, method: str) -> np.ndarray:
        """Apply a specific augmentation method to an MFCC sample."""
        if method == 'add_noise':
            return self.augmenter.add_noise(mfcc, noise_factor=0.01)
        elif method == 'time_shift':
            return self.augmenter.time_shift(mfcc, shift_factor=0.1)
        elif method == 'pitch_shift':
            return self.augmenter.pitch_shift(mfcc, shift_factor=0.1)
        elif method == 'time_stretch':
            return self.augmenter.time_stretch(mfcc, stretch_factor=0.1)
        elif method == 'frequency_mask':
            return self.augmenter.frequency_mask(mfcc, mask_factor=0.1)
        else:
            raise ValueError(f"Unknown augmentation method: {method}")
    
    def balance_dataset(self, features: List[np.ndarray], labels: List[str],
                       target_ratio: float = 0.1) -> Tuple[List[np.ndarray], List[str]]:
        """
        Balance the dataset by augmenting underrepresented classes.
        
        Args:
            features: List of MFCC feature arrays
            labels: List of corresponding labels
            target_ratio: Target ratio for minority classes
            
        Returns:
            Tuple of (augmented_features, augmented_labels)
        """
        print(f"\nBalancing dataset with target ratio: {target_ratio}")
        
        # Analyze current distribution
        class_counts = self.analyze_class_distribution(labels)
        target_samples = self.calculate_target_samples(class_counts, target_ratio)
        
        # Group samples by class
        class_samples = {}
        for mfcc, label in zip(features, labels):
            if label not in class_samples:
                class_samples[label] = []
            class_samples[label].append(mfcc)
        
        # Augment underrepresented classes
        augmented_features = list(features)
        augmented_labels = list(labels)
        
        for genre, current_count in class_counts.items():
            target_count = target_samples[genre]
            needed_samples = target_count - current_count
            
            if needed_samples > 0:
                print(f"\nAugmenting {genre}: {current_count} -> {target_count} samples (+{needed_samples})")
                
                # Get existing samples for this class
                existing_samples = class_samples[genre]
                
                # Generate augmented samples
                for i in range(needed_samples):
                    # Select a random existing sample
                    base_sample = existing_samples[np.random.randint(len(existing_samples))]
                    
                    # Apply random augmentation
                    method = np.random.choice(self.augmentation_methods)
                    augmented_sample = self.augment_sample(base_sample, method)
                    
                    augmented_features.append(augmented_sample)
                    augmented_labels.append(genre)
        
        print(f"\nBalancing complete!")
        print(f"Original samples: {len(features)}")
        print(f"Augmented samples: {len(augmented_features)}")
        print(f"Added samples: {len(augmented_features) - len(features)}")
        
        return augmented_features, augmented_labels
    
    def save_balanced_dataset(self, features: List[np.ndarray], labels: List[str],
                            output_path: str):
        """Save the balanced dataset to a JSON file."""
        print(f"\nSaving balanced dataset to {output_path}...")
        
        # Convert to the new format expected by the trainer
        features_list = [mfcc.tolist() for mfcc in features]
        labels_list = labels
        
        # Create genre mapping
        unique_labels = sorted(set(labels))
        genre_mapping = {i: label for i, label in enumerate(unique_labels)}
        
        # Convert string labels to integer indices
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        labels_int = [label_to_idx[label] for label in labels]
        
        data = {
            'features': features_list,
            'labels': labels_int,
            'mapping': genre_mapping
        }
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(features_list)} samples to {output_path}")
        
        # Show final distribution
        final_counts = Counter(labels_int)
        print("\nFinal Class Distribution:")
        print("-" * 40)
        for genre, count in sorted(final_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(labels_int)) * 100
            print(f"{str(genre):20s}: {count:4d} samples ({percentage:5.1f}%)")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Balance dataset by augmenting underrepresented classes')
    parser.add_argument('--input', required=True, help='Input JSON file path')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    parser.add_argument('--target-ratio', type=float, default=0.1, 
                       help='Target ratio for minority classes (default: 0.1)')
    parser.add_argument('--methods', nargs='+', 
                       default=['add_noise', 'time_shift', 'pitch_shift', 'time_stretch', 'frequency_mask'],
                       help='Augmentation methods to use')
    
    args = parser.parse_args()
    
    # Create balancer
    balancer = DatasetBalancer(augmentation_methods=args.methods)
    
    # Load dataset
    features, labels = balancer.load_dataset(args.input)
    
    # Balance dataset
    balanced_features, balanced_labels = balancer.balance_dataset(
        features, labels, target_ratio=args.target_ratio
    )
    
    # Save balanced dataset
    balancer.save_balanced_dataset(balanced_features, balanced_labels, args.output)


if __name__ == "__main__":
    main()
