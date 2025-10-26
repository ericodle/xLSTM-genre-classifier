#!/usr/bin/env python3
"""
Test individual model branches to identify which components work best.
"""

import argparse
import json
import logging
import os
import sys
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from multimodal.model_branches import SpectralCNNBranch, TemporalRNNBranch, StatisticalFCBranch
from multimodal.feature_extractor import MultimodalFeatureExtractor, MultimodalFeatures
from core.constants import *

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


class BranchDataset(Dataset):
    """Dataset for individual branch testing."""
    
    def __init__(self, features: List[MultimodalFeatures], labels: List[str], class_to_idx: dict, branch_type: str):
        self.features = features
        self.labels = [class_to_idx[label] for label in labels]
        self.class_to_idx = class_to_idx
        self.branch_type = branch_type
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        
        if self.branch_type == 'spectral':
            # Process spectral features for CNN
            spectral_tensor = self.process_spectral_features(feature)
            return spectral_tensor, self.labels[idx]
        elif self.branch_type == 'temporal':
            # Process temporal features for RNN
            temporal_tensor = self.process_temporal_features(feature)
            return temporal_tensor, self.labels[idx]
        elif self.branch_type == 'statistical':
            # Process statistical features for FC
            statistical_tensor = self.process_statistical_features(feature)
            return statistical_tensor, self.labels[idx]
        else:
            raise ValueError(f"Unknown branch type: {self.branch_type}")
    
    def process_spectral_features(self, feature: MultimodalFeatures) -> torch.Tensor:
        """Process spectral features for CNN branch."""
        # Stack features as separate channels instead of concatenating
        features_list = [
            feature.mel_spectrogram,           # (128, time)
            feature.chroma,                    # (12, time)  
            feature.spectral_centroid,         # (1, time)
            feature.spectral_rolloff,          # (1, time)
            feature.spectral_contrast,         # (7, time)
            feature.zero_crossing_rate         # (1, time)
        ]
        
        # Pad shorter features to match the longest (mel_spectrogram)
        max_height = max(f.shape[0] for f in features_list)
        padded_features = []
        
        for f in features_list:
            if f.shape[0] < max_height:
                # Pad with zeros
                pad_height = max_height - f.shape[0]
                f = np.pad(f, ((0, pad_height), (0, 0)), mode='constant')
            padded_features.append(f)
        
        # Stack as channels
        features = np.stack(padded_features, axis=0)  # (6, max_height, time)
        
        # Pad or truncate to fixed length
        target_length = 1000  # Fixed length for all samples
        if features.shape[2] > target_length:
            features = features[:, :, :target_length]  # Truncate
        else:
            # Pad with zeros
            pad_width = target_length - features.shape[2]
            features = np.pad(features, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
        
        # Convert to tensor (batch dimension added by collate)
        features_tensor = torch.FloatTensor(features)  # (6, max_height, 1000)
        
        return features_tensor
    
    def process_temporal_features(self, feature: MultimodalFeatures) -> torch.Tensor:
        """Process temporal features for RNN branch."""
        # Concatenate MFCC and derivatives
        temporal_features = np.concatenate([
            feature.mfcc,        # (13, time)
            feature.delta_mfcc,  # (13, time)
            feature.delta2_mfcc # (13, time)
        ], axis=0)  # Result: (39, time)
        
        # Pad or truncate to fixed length
        target_length = 1000  # Fixed length for all samples
        if temporal_features.shape[1] > target_length:
            temporal_features = temporal_features[:, :target_length]  # Truncate
        else:
            # Pad with zeros
            pad_width = target_length - temporal_features.shape[1]
            temporal_features = np.pad(temporal_features, ((0, 0), (0, pad_width)), mode='constant')
        
        # Convert to tensor and transpose for RNN (time, features)
        temporal_tensor = torch.FloatTensor(temporal_features).transpose(0, 1)  # (1000, 39)
        
        return temporal_tensor
    
    def process_statistical_features(self, feature: MultimodalFeatures) -> torch.Tensor:
        """Process statistical features for FC branch."""
        # Combine all statistical features into a single vector
        statistical_features = []
        
        # Add scalar features
        statistical_features.append(feature.tempo)
        statistical_features.append(feature.harmonic_percussive_ratio)
        
        # Add array features (flatten them)
        statistical_features.extend(feature.beat_frames.flatten())
        statistical_features.extend(feature.onset_strength.flatten())
        statistical_features.extend(feature.spectral_bandwidth.flatten())
        statistical_features.extend(feature.spectral_flatness.flatten())
        
        # Convert to numpy array first, then to tensor
        statistical_array = np.array(statistical_features, dtype=np.float32)
        
        # Pad or truncate to fixed length
        target_length = 4000  # Fixed length for all samples
        if len(statistical_array) > target_length:
            statistical_array = statistical_array[:target_length]  # Truncate
        else:
            # Pad with zeros
            pad_width = target_length - len(statistical_array)
            statistical_array = np.pad(statistical_array, (0, pad_width), mode='constant')
        
        return torch.FloatTensor(statistical_array)


def branch_collate_fn(batch):
    """Custom collate function for branch testing."""
    features, labels = zip(*batch)
    if isinstance(features[0], torch.Tensor):
        features = torch.stack(features)
    else:
        features = torch.stack(features)
    return features, torch.LongTensor(labels)


class BranchTrainer:
    """Trainer for individual branches."""
    
    def __init__(self, model: nn.Module, device: torch.device, class_names: List[str]):
        self.model = model
        self.device = device
        self.class_names = class_names
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              test_loader: DataLoader, epochs: int = 50) -> float:
        """Train the branch model."""
        
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0.0
        patience_counter = 0
        patience = 10
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Final evaluation
        test_acc, report, cm = self.evaluate(test_loader, criterion)
        logger.info(f"Test Accuracy: {test_acc:.2f}%")
        
        return test_acc
    
    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer, 
                   criterion: nn.Module) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for features, labels in train_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        return total_loss / len(train_loader), 100. * correct / total
    
    def validate(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return total_loss / len(val_loader), 100. * correct / total
    
    def evaluate(self, test_loader: DataLoader, criterion: nn.Module) -> Tuple[float, str, np.ndarray]:
        """Evaluate the model on test set."""
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(features)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = 100 * sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_labels)
        
        # Classification report
        report = classification_report(all_labels, all_predictions, target_names=self.class_names)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        return accuracy, report, cm


def load_multimodal_data(data_path: str) -> Tuple[List[MultimodalFeatures], List[str]]:
    """Load multimodal features from dataset."""
    
    # Load JSON data
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    features_list = []
    labels = []
    
    # Extract features for each sample
    for i, (features, label) in enumerate(zip(data['features'], data['labels'])):
        if i % 100 == 0:
            print(f"Processing sample {i+1}/{len(data['features'])}")
        
        # Convert features back to MultimodalFeatures object
        multimodal_features = MultimodalFeatures(
            # Spectral features
            mel_spectrogram=np.array(features['mel_spectrogram']),
            chroma=np.array(features['chroma']),
            spectral_centroid=np.array(features['spectral_centroid']),
            spectral_rolloff=np.array(features['spectral_rolloff']),
            spectral_contrast=np.array(features['spectral_contrast']),
            zero_crossing_rate=np.array(features['zero_crossing_rate']),
            
            # Temporal features
            mfcc=np.array(features['mfcc']),
            delta_mfcc=np.array(features['delta_mfcc']),
            delta2_mfcc=np.array(features['delta2_mfcc']),
            
            # Statistical features
            tempo=features['tempo'],
            beat_frames=np.array(features['beat_frames']),
            onset_strength=np.array(features['onset_strength']),
            harmonic_percussive_ratio=features['harmonic_percussive_ratio'],
            spectral_bandwidth=np.array(features['spectral_bandwidth']),
            spectral_flatness=np.array(features['spectral_flatness'])
        )
        
        features_list.append(multimodal_features)
        labels.append(label)
    
    return features_list, labels


def main():
    parser = argparse.ArgumentParser(description='Test individual model branches')
    parser.add_argument('--data', type=str, required=True, help='Path to multimodal features JSON file')
    parser.add_argument('--output', type=str, required=True, help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data from {args.data}")
    features_list, labels = load_multimodal_data(args.data)
    
    # Get unique classes
    class_names = sorted(list(set(labels)))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    num_classes = len(class_names)
    
    logger.info(f"Found {num_classes} classes: {class_names}")
    logger.info(f"Total samples: {len(features_list)}")
    
    # Split data
    from sklearn.model_selection import train_test_split
    
    # First split: train+val vs test
    train_val_features, test_features, train_val_labels, test_labels = train_test_split(
        features_list, labels, test_size=0.15, random_state=42, stratify=labels
    )
    
    # Second split: train vs val
    train_features, val_features, train_labels, val_labels = train_test_split(
        train_val_features, train_val_labels, test_size=0.18, random_state=42, stratify=train_val_labels
    )
    
    logger.info(f"Data split - Train: {len(train_features)}, Val: {len(val_features)}, Test: {len(test_features)}")
    
    # Test each branch
    branch_results = {}
    
    for branch_type in ['spectral', 'temporal', 'statistical']:
        logger.info(f"\n=== Testing {branch_type.upper()} branch ===")
        
        # Create dataset
        train_dataset = BranchDataset(train_features, train_labels, class_to_idx, branch_type)
        val_dataset = BranchDataset(val_features, val_labels, class_to_idx, branch_type)
        test_dataset = BranchDataset(test_features, test_labels, class_to_idx, branch_type)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                 collate_fn=branch_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                               collate_fn=branch_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                                collate_fn=branch_collate_fn)
        
        # Create model
        if branch_type == 'spectral':
            model = SpectralCNNBranch(num_classes=num_classes, dropout=0.3).to(device)
        elif branch_type == 'temporal':
            model = TemporalRNNBranch(input_dim=39, num_classes=num_classes, dropout=0.3).to(device)
        elif branch_type == 'statistical':
            # Get input dimension from first sample
            sample_tensor = train_dataset[0][0]
            input_dim = sample_tensor.shape[0]
            model = StatisticalFCBranch(input_dim=input_dim, num_classes=num_classes, dropout=0.3).to(device)
        
        logger.info(f"{branch_type.capitalize()} model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Create trainer
        trainer = BranchTrainer(model, device, class_names)
        
        # Train model
        test_acc = trainer.train(train_loader, val_loader, test_loader, epochs=args.epochs)
        branch_results[branch_type] = test_acc
    
    # Print results
    logger.info("\n=== BRANCH COMPARISON RESULTS ===")
    for branch_type, accuracy in branch_results.items():
        logger.info(f"{branch_type.capitalize()} branch: {accuracy:.2f}%")
    
    # Save results
    results = {
        "branch_results": branch_results,
        "class_names": class_names,
        "num_classes": num_classes
    }
    
    with open(os.path.join(args.output, 'branch_comparison.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
