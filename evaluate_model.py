#!/usr/bin/env python3
"""
Simple model evaluation script for GenreDiscern.
This script evaluates a trained model against the entire dataset.
"""

import sys
import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from scipy.stats import ks_2samp

# Add src directory to path
sys.path.insert(0, 'src')

def load_model(model_path):
    """Load a trained model from checkpoint or ONNX file."""
    # Check if it's an ONNX file
    if model_path.endswith('.onnx'):
        try:
            import onnxruntime as ort
            print("Loading ONNX model...")
            session = ort.InferenceSession(model_path)
            
            # Detect model type from ONNX model structure
            model_type = detect_onnx_model_type(session)
            print(f"Detected {model_type} model from ONNX")
            
            return session, model_type
        except ImportError:
            print("ONNX Runtime not available, falling back to PyTorch")
        except Exception as e:
            print(f"ONNX loading failed: {e}")
            print("Falling back to PyTorch")
    
    # Try to load as PyTorch checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # It's a checkpoint, need to create model and load state dict
        from src.models import get_model
        
        # Try to detect model type from the state dict keys
        state_dict = checkpoint['model_state_dict']
        
        # Check for different model architectures
        if 'rnn.weight_ih_l0' in state_dict:
            # GRU or LSTM model
            # Calculate dimensions from state dict
            input_dim = state_dict['rnn.weight_ih_l0'].shape[1]  # 13
            # For GRU: weight_ih_l0 has shape [3*hidden_dim, input_dim] (3 gates)
            # For LSTM: weight_ih_l0 has shape [4*hidden_dim, input_dim] (4 gates)
            # Let's check which one it is
            total_gates = state_dict['rnn.weight_ih_l0'].shape[0]
            if total_gates % 3 == 0:
                # GRU model
                hidden_dim = total_gates // 3
                model_type = 'GRU'
            elif total_gates % 4 == 0:
                # LSTM model
                hidden_dim = total_gates // 4
                model_type = 'LSTM'
            else:
                # Unknown, assume GRU
                hidden_dim = total_gates // 3
                model_type = 'GRU'
            
            output_dim = state_dict['fc.weight'].shape[0]  # 10
            
            try:
                if model_type == 'GRU':
                    model = get_model('GRU', input_dim=input_dim, hidden_dim=hidden_dim, num_layers=1, output_dim=output_dim, dropout=0.0)
                    print(f"Created GRU model with input_dim={input_dim}, hidden_dim={hidden_dim}, output_dim={output_dim}")
                else:
                    model = get_model('LSTM', input_dim=input_dim, hidden_dim=hidden_dim, num_layers=1, output_dim=output_dim, dropout=0.0)
                    print(f"Created LSTM model with input_dim={input_dim}, hidden_dim={hidden_dim}, output_dim={output_dim}")
            except Exception as e:
                print(f"Error creating {model_type} model: {e}")
                # Fallback to FC model
                model = get_model('FC')
                print("Fallback to FC model")
        elif 'conv_layers.0.weight' in state_dict:
            # CNN model - always use the new architecture
            print("Detected CNN model - using new dynamic architecture")
            model = get_model('CNN')
        else:
            # Default to FC model
            model = get_model('FC')
            print("Detected FC model")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model from checkpoint")
    else:
        # Assume it's a direct model
        model = checkpoint
        print("Loaded direct model")
    
    # Only call eval() for PyTorch models
    if not hasattr(model, 'session'):
        model.eval()
    return model

def detect_onnx_model_type(session):
    """Detect model type from ONNX model structure."""
    # Get model metadata
    meta = session.get_modelmeta()
    
    # Check input shape to determine model type
    input_shape = session.get_inputs()[0].shape
    
    if len(input_shape) == 4:
        # 4D input: (batch, channels, height, width) -> CNN
        return 'CNN'
    elif len(input_shape) == 3:
        # 3D input: (batch, time_steps, features) -> RNN (LSTM/GRU)
        return 'RNN'
    elif len(input_shape) == 2:
        # 2D input: (batch, features) -> FC
        return 'FC'
    else:
        # Default to FC for unknown shapes
        return 'FC'

def create_onnx_wrapper(session, model_type):
    """Create a PyTorch-like wrapper for ONNX models."""
    class ONNXModelWrapper:
        def __init__(self, session, model_type):
            self.session = session
            self.model_type = model_type
            self.eval = lambda: None  # ONNX models are always in eval mode
            
        def __call__(self, x):
            # Convert PyTorch tensor to numpy and run inference
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            return torch.from_numpy(self.session.run(['output'], {'input': x})[0])
            
    return ONNXModelWrapper(session, model_type)



def load_mfcc_data(json_path):
    """Load MFCC data from the JSON file."""
    with open(json_path, 'r') as f:
        mfcc_data = json.load(f)
    
    # Extract features and labels from the file path structure
    features = []
    labels = []
    mapping = []
    
    for file_path, data_dict in mfcc_data.items():
        # Extract genre from file path (e.g., "reggae/reggae.00044.wav" -> "reggae")
        genre = file_path.split('/')[0]
        
        if genre not in mapping:
            mapping.append(genre)
        
        genre_idx = mapping.index(genre)
        features.append(data_dict['mfcc'])
        labels.append(genre_idx)
    
    # Convert to numpy arrays
    features = np.array(features)
    labels = np.array(labels)
    
    return features, labels, mapping

def preprocess_features(features, flatten_for_rnn=False, is_cnn=False):
    """Preprocess features (normalization, reshaping, etc.)."""
    # Ensure features are 2D
    if len(features.shape) == 3:
        if flatten_for_rnn:
            # If features are 3D (samples, time_steps, mfcc_coeffs)
            # Reshape to 2D (samples, time_steps * mfcc_coeffs)
            features = features.reshape(features.shape[0], -1)
        elif is_cnn:
            # For CNN models, reshape to 4D (batch, channels, height, width)
            # (samples, time_steps, mfcc_coeffs) -> (samples, 1, time_steps, mfcc_coeffs)
            features = features.reshape(features.shape[0], 1, features.shape[1], features.shape[2])
        # For RNN models, keep 3D format (samples, time_steps, mfcc_coeffs)
    
    # Normalize features
    if len(features.shape) == 3:
        # 3D features: normalize each sample independently
        for i in range(features.shape[0]):
            features[i] = (features[i] - np.mean(features[i])) / (np.std(features[i]) + 1e-8)
    elif len(features.shape) == 4:
        # 4D features (CNN): normalize each sample independently
        for i in range(features.shape[0]):
            features[i] = (features[i] - np.mean(features[i])) / (np.std(features[i]) + 1e-8)
    else:
        # 2D features: normalize across samples
        features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
    
    return features

class SimpleDataset(torch.utils.data.Dataset):
    """Simple dataset for evaluation."""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def evaluate_model(model, test_loader, class_names):
    """Evaluate the model and return results."""
    # Only call eval() for PyTorch models
    if not hasattr(model, 'eval'):
        pass  # ONNX models don't have eval()
    else:
        model.eval()
    
    y_true = []
    y_pred = []
    y_probs = []
    
    with torch.no_grad():
        for data, target in test_loader:
            # Forward pass - handle both PyTorch and ONNX models
            if hasattr(model, 'model_type'):  # ONNX model wrapper
                # Convert to numpy for ONNX
                data_np = data.detach().cpu().numpy()
                output = model.session.run(['output'], {'input': data_np})[0]
                output = torch.from_numpy(output)
            else:
                # PyTorch model
                output = model(data)
            
            # Get probabilities and predictions
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1)
            
            # Store results
            y_true.append(target.numpy())
            y_pred.append(pred.numpy())
            y_probs.append(probs.numpy())
    
    # Concatenate all batches
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_probs = np.concatenate(y_probs)
    
    # Calculate metrics
    accuracy = (y_true == y_pred).mean()
    
    # Classification report
    classification_rep = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # ROC AUC scores
    roc_auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
    
    # KS test statistics
    ks_stats = []
    for class_idx in range(y_probs.shape[1]):
        class_true = y_probs[y_true == class_idx, class_idx]
        class_all = y_probs[:, class_idx]
        
        if len(class_true) > 0:
            ks_stat, _ = ks_2samp(class_true, class_all)
            ks_stats.append(ks_stat)
        else:
            ks_stats.append(0.0)
    
    return {
        'accuracy': accuracy,
        'classification_report': classification_rep,
        'confusion_matrix': cm,
        'roc_auc': roc_auc,
        'ks_stats': ks_stats,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_probs': y_probs
    }

def plot_confusion_matrix(cm, class_names, output_path):
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_ks_curves(results, class_names, output_path):
    """Plot KS curves."""
    plt.figure(figsize=(10, 8))
    
    y_true = results['y_true']
    y_probs = results['y_probs']
    
    for class_idx in range(y_probs.shape[1]):
        # Get probabilities for samples that actually belong to this class
        class_true = y_probs[y_true == class_idx, class_idx]
        
        if len(class_true) > 0:
            class_name = class_names[class_idx]
            ks_stat = results['ks_stats'][class_idx]
            
            # Sort probabilities for plotting
            sorted_probs = np.sort(class_true)
            plt.plot(sorted_probs, label=f'{class_name} (KS = {ks_stat:.3f})')
    
    plt.xlabel('Sorted Class Probabilities')
    plt.ylabel('CDF')
    plt.title('KS Curves')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_results(results, output_path):
    """Save evaluation results to file."""
    # Save metrics to text file
    metrics_file = os.path.join(output_path, 'evaluation_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"ROC AUC: {results['roc_auc']:.4f}\n\n")
        
        f.write("KS Statistics:\n")
        for i, ks_stat in enumerate(results['ks_stats']):
            f.write(f"Class {i}: {ks_stat:.4f}\n")
        
        f.write("\nClassification Report:\n")
        f.write(classification_report(results['y_true'], results['y_pred'], 
                                   target_names=[f"Class_{i}" for i in range(len(results['ks_stats']))]))
    
    print(f"Results saved to {metrics_file}")

def main():
    if len(sys.argv) != 4:
        print("Usage: python evaluate_model.py <model_path> <data_path> <output_dir>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    data_path = sys.argv[2]
    output_dir = sys.argv[3]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading model...")
    result = load_model(model_path)
    
    # Handle different return types
    if isinstance(result, tuple):
        # ONNX model returned (session, model_type)
        model, model_type = result
        print(f"Using ONNX {model_type} model for evaluation")
        model = create_onnx_wrapper(model, model_type)
        is_onnx = True
    else:
        # PyTorch model returned
        model = result
        is_onnx = False
    
    print("Loading data...")
    features, labels, mapping = load_mfcc_data(data_path)
    
    print("Preprocessing features...")
    # Determine preprocessing based on model type
    if is_onnx:
        if model.model_type == 'CNN':
            features = preprocess_features(features, flatten_for_rnn=False, is_cnn=True)
        elif model.model_type == 'RNN':
            features = preprocess_features(features, flatten_for_rnn=False, is_cnn=False)
        else:  # FC model
            features = preprocess_features(features, flatten_for_rnn=True, is_cnn=False)
    else:
        # PyTorch model - use existing logic
        features = preprocess_features(features, flatten_for_rnn=False, is_cnn=False)
    
    print("Creating dataset...")
    dataset = SimpleDataset(features, labels)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # GTZAN genre names (standard order)
    class_names = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    print("Evaluating model...")
    results = evaluate_model(model, test_loader, class_names)
    
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"ROC AUC: {results['roc_auc']:.4f}")
    
    print("Generating plots...")
    plot_confusion_matrix(results['confusion_matrix'], class_names, 
                         os.path.join(output_dir, 'confusion_matrix.png'))
    plot_ks_curves(results, class_names, os.path.join(output_dir, 'ks_curves.png'))
    
    print("Saving results...")
    save_results(results, output_dir)
    
    print("Evaluation completed successfully!")

if __name__ == "__main__":
    main() 