#!/usr/bin/env python3
"""
Improved multimodal training script with better hyperparameters and debugging.
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

from multimodal.multimodal_model import MultimodalModel
from multimodal.feature_extractor import MultimodalFeatureExtractor, MultimodalFeatures
from core.constants import *

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


class MultimodalDataset(Dataset):
    """Dataset for multimodal features."""
    
    def __init__(self, features: List[MultimodalFeatures], labels: List[str], class_to_idx: dict):
        self.features = features
        self.labels = [class_to_idx[label] for label in labels]
        self.class_to_idx = class_to_idx
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def multimodal_collate_fn(batch):
    """Custom collate function for multimodal features."""
    features, labels = zip(*batch)
    return list(features), torch.LongTensor(labels)


class ImprovedMultimodalTrainer:
    """Improved trainer with better hyperparameters and debugging."""
    
    def __init__(self, model: MultimodalModel, device: torch.device, class_names: List[str]):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.best_val_acc = 0.0
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              test_loader: DataLoader, output_dir: str, epochs: int = 100):
        """Train the model with improved hyperparameters."""
        
        # Use better hyperparameters
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=0.001,  # Lower learning rate
            weight_decay=0.01,  # Higher weight decay
            betas=(0.9, 0.999)
        )
        
        # Use cosine annealing scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        
        criterion = nn.CrossEntropyLoss()
        
        # Early stopping with more patience
        best_val_acc = 0.0
        patience_counter = 0
        patience = 20  # Increased patience
        
        logger.info(f"Starting training for {epochs} epochs...")
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        logger.info(f"Batch size: {train_loader.batch_size}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion, epoch)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            # Update scheduler
            scheduler.step()
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
                logger.info(f"New best validation accuracy: {val_acc:.2f}%")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model and evaluate
        self.model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth')))
        logger.info(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
        
        # Final evaluation
        test_acc, report, cm = self.evaluate(test_loader, criterion)
        logger.info(f"Test Accuracy: {test_acc:.2f}%")
        
        # Save results
        self.save_results(output_dir, test_acc, report, cm)
        
        # Plot training history
        self.plot_training_history(output_dir)
        
        return test_acc
    
    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer, 
                   criterion: nn.Module, epoch: int) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (features, labels) in enumerate(train_loader):
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            
            # Process batch
            batch_outputs = []
            for feature in features:
                feature = feature.to(self.device)
                output = self.model(feature)
                batch_outputs.append(output)
            
            # Stack outputs and squeeze extra dimension
            outputs = torch.stack(batch_outputs).squeeze(1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, "
                           f"Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%")
        
        return total_loss / len(train_loader), 100. * correct / total
    
    def validate(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                labels = labels.to(self.device)
                
                batch_outputs = []
                for feature in features:
                    feature = feature.to(self.device)
                    output = self.model(feature)
                    batch_outputs.append(output)
                
                outputs = torch.stack(batch_outputs).squeeze(1)
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
                labels = labels.to(self.device)
                
                batch_outputs = []
                for feature in features:
                    feature = feature.to(self.device)
                    output = self.model(feature)
                    batch_outputs.append(output)
                
                outputs = torch.stack(batch_outputs).squeeze(1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = 100 * sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_labels)
        
        # Classification report
        report = classification_report(all_labels, all_predictions, target_names=self.class_names)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        return accuracy, report, cm
    
    def save_results(self, output_dir: str, test_acc: float, report: str, cm: np.ndarray):
        """Save evaluation results."""
        results = {
            "accuracy": test_acc,
            "classification_report": report,
            "confusion_matrix": cm.tolist()
        }
        
        with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
    
    def plot_training_history(self, output_dir: str):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Val Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accs, label='Train Acc', color='blue')
        ax2.plot(self.val_accs, label='Val Acc', color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()


def load_multimodal_data(data_path: str, feature_extractor: MultimodalFeatureExtractor) -> Tuple[List[MultimodalFeatures], List[str]]:
    """Load and extract multimodal features from dataset."""
    
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
    parser = argparse.ArgumentParser(description='Train improved multimodal model')
    parser.add_argument('--data', type=str, required=True, help='Path to multimodal features JSON file')
    parser.add_argument('--output', type=str, required=True, help='Output directory for model and results')
    parser.add_argument('--fusion', type=str, default='concat', choices=['concat', 'attention', 'weighted'],
                       help='Fusion method')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
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
    feature_extractor = MultimodalFeatureExtractor()
    features_list, labels = load_multimodal_data(args.data, feature_extractor)
    
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
    
    # Create datasets
    train_dataset = MultimodalDataset(train_features, train_labels, class_to_idx)
    val_dataset = MultimodalDataset(val_features, val_labels, class_to_idx)
    test_dataset = MultimodalDataset(test_features, test_labels, class_to_idx)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             collate_fn=multimodal_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                           collate_fn=multimodal_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                             collate_fn=multimodal_collate_fn)
    
    # Create model
    model = MultimodalModel(
        num_classes=num_classes,
        fusion_method=args.fusion,
        device=device,
        dropout=0.3  # Higher dropout for regularization
    )
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create trainer
    trainer = ImprovedMultimodalTrainer(model, device, class_names)
    
    # Train model
    test_acc = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        output_dir=args.output,
        epochs=args.epochs
    )
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output, 'final_model.pth'))
    
    logger.info(f"Results saved to {args.output}")
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
