"""
Training script for multimodal music genre classification.
Handles multimodal feature extraction and training of specialized branches.
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.constants import (
    DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, DEFAULT_MAX_EPOCHS,
    DEFAULT_WEIGHT_DECAY, DEFAULT_OPTIMIZER, DEFAULT_LOSS_FUNCTION,
    DEFAULT_LR_SCHEDULER, DEFAULT_CLASS_WEIGHT, DEFAULT_RANDOM_SEED,
    DEFAULT_NUM_WORKERS, DEFAULT_PIN_MEMORY, GTZAN_GENRES
)
from multimodal.feature_extractor import MultimodalFeatureExtractor, MultimodalFeatures
from multimodal.multimodal_model import MultimodalModel


class MultimodalDataset(Dataset):
    """Dataset class for multimodal features."""
    
    def __init__(self, features_list: List[MultimodalFeatures], labels: List[int]):
        self.features_list = features_list
        self.labels = labels
    
    def __len__(self):
        return len(self.features_list)
    
    def __getitem__(self, idx):
        return self.features_list[idx], self.labels[idx]


class MultimodalTrainer:
    """Trainer for multimodal music genre classification."""
    
    def __init__(
        self,
        model: MultimodalModel,
        device: str = "auto",
        logger: Optional[logging.Logger] = None
    ):
        self.model = model
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger or logging.getLogger(__name__)
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        epoch: int
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (features_list, labels) in enumerate(train_loader):
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            
            # Process each sample in the batch
            batch_outputs = []
            for features in features_list:
                outputs = self.model(features)
                batch_outputs.append(outputs)
            
            # Stack outputs for the batch and squeeze the extra dimension
            outputs = torch.stack(batch_outputs).squeeze(1)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                self.logger.info(
                    f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                    f'Loss: {loss.item():.4f}, Acc: {100 * correct / total:.2f}%'
                )
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features_list, labels in val_loader:
                labels = labels.to(self.device)
                
                # Process each sample in the batch
                batch_outputs = []
                for features in features_list:
                    outputs = self.model(features)
                    batch_outputs.append(outputs)
                
                # Stack outputs for the batch and squeeze the extra dimension
                outputs = torch.stack(batch_outputs).squeeze(1)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = DEFAULT_MAX_EPOCHS,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        optimizer_name: str = DEFAULT_OPTIMIZER,
        lr_scheduler: bool = DEFAULT_LR_SCHEDULER,
        class_weight: str = DEFAULT_CLASS_WEIGHT,
        output_dir: str = "outputs/multimodal"
    ):
        """Train the multimodal model."""
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup optimizer
        if optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Setup loss function
        if class_weight == "auto":
            # Calculate class weights
            class_counts = {}
            for _, labels in train_loader:
                for label in labels:
                    class_counts[label.item()] = class_counts.get(label.item(), 0) + 1
            
            total_samples = sum(class_counts.values())
            class_weights = [total_samples / (len(class_counts) * count) for count in class_counts.values()]
            class_weights = torch.FloatTensor(class_weights).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Setup learning rate scheduler
        scheduler = None
        if lr_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )
        
        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        patience = 10
        
        self.logger.info(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion, epoch)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            # Update learning rate
            if scheduler:
                scheduler.step(val_loss)
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Log progress
            self.logger.info(
                f'Epoch {epoch+1}/{num_epochs}: '
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%'
            )
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
                self.logger.info(f'New best model saved! Val Acc: {val_acc:.2f}%')
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                self.logger.info(f'Early stopping at epoch {epoch+1}')
                break
        
        # Save final model
        torch.save(self.model.state_dict(), os.path.join(output_dir, 'final_model.pth'))
        
        # Plot training history
        self.plot_training_history(output_dir)
        
        self.logger.info(f'Training completed! Best validation accuracy: {best_val_acc:.2f}%')
    
    def plot_training_history(self, output_dir: str):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def evaluate(
        self,
        test_loader: DataLoader,
        class_names: List[str] = GTZAN_GENRES,
        output_dir: str = "outputs/multimodal"
    ):
        """Evaluate the model on test set."""
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for features_list, labels in test_loader:
                labels = labels.to(self.device)
                
                # Process each sample in the batch
                batch_outputs = []
                for features in features_list:
                    outputs = self.model(features)
                    batch_outputs.append(outputs)
                
                # Stack outputs for the batch and squeeze the extra dimension
                outputs = torch.stack(batch_outputs).squeeze(1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = 100 * sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_labels)
        
        # Classification report
        report = classification_report(all_labels, all_predictions, target_names=class_names)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save results
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
        
        with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f'Test Accuracy: {accuracy:.2f}%')
        self.logger.info(f'Results saved to {output_dir}')
        
        return results


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
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train multimodal music genre classifier')
    parser.add_argument('--data', required=True, help='Path to training data JSON file')
    parser.add_argument('--output', required=True, help='Output directory for models and results')
    parser.add_argument('--epochs', type=int, default=DEFAULT_MAX_EPOCHS, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=DEFAULT_LEARNING_RATE, help='Learning rate')
    parser.add_argument('--fusion', choices=['attention', 'concat', 'weighted'], default='attention', help='Fusion method')
    parser.add_argument('--device', default='auto', help='Device to use (cpu/cuda/auto)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize feature extractor
    feature_extractor = MultimodalFeatureExtractor()
    
    # Load data
    logger.info(f"Loading data from {args.data}")
    features_list, labels = load_multimodal_data(args.data, feature_extractor)
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    # Split data
    train_features, temp_features, train_labels, temp_labels = train_test_split(
        features_list, encoded_labels, test_size=0.3, random_state=DEFAULT_RANDOM_SEED
    )
    val_features, test_features, val_labels, test_labels = train_test_split(
        temp_features, temp_labels, test_size=0.5, random_state=DEFAULT_RANDOM_SEED
    )
    
    logger.info(f"Data split - Train: {len(train_features)}, Val: {len(val_features)}, Test: {len(test_features)}")
    
    # Create datasets
    train_dataset = MultimodalDataset(train_features, train_labels)
    val_dataset = MultimodalDataset(val_features, val_labels)
    test_dataset = MultimodalDataset(test_features, test_labels)
    
    # Create data loaders with custom collate function
    def multimodal_collate_fn(batch):
        """Custom collate function for multimodal features."""
        features, labels = zip(*batch)
        return list(features), torch.LongTensor(labels)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             collate_fn=multimodal_collate_fn, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                           collate_fn=multimodal_collate_fn, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                            collate_fn=multimodal_collate_fn, num_workers=0, pin_memory=False)
    
    # Create model
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalModel(
        num_classes=len(label_encoder.classes_),
        fusion_method=args.fusion,
        device=device
    )
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = MultimodalTrainer(model, device=device, logger=logger)
    
    # Train model
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        output_dir=args.output
    )
    
    # Evaluate model
    trainer.evaluate(test_loader, label_encoder.classes_, args.output)
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
