#!/usr/bin/env python3
"""
Simple model evaluation script for GenreDiscern.
This script evaluates a trained model against the entire dataset.
"""

import sys
import os
import json
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from scipy.stats import ks_2samp

# Suppress sklearn warnings about undefined metrics
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.metrics')

# Add src directory to path
sys.path.insert(0, 'src')

def load_model(model_path):
    """Load a trained model from ONNX file."""
    # Ensure we're working with ONNX files
    if not model_path.endswith('.onnx'):
        # Try to find corresponding ONNX file
        onnx_path = model_path.replace('.pth', '.onnx')
        if os.path.exists(onnx_path):
            model_path = onnx_path
        else:
            raise FileNotFoundError(f"ONNX model not found. Expected: {onnx_path}")
    
    try:
        import onnxruntime as ort
        print("Loading ONNX model...")
        session = ort.InferenceSession(model_path)
        
        # Detect model type from ONNX model structure
        model_type = detect_onnx_model_type(session)
        print(f"Detected {model_type} model from ONNX")
        
        return session, model_type
    except ImportError:
        raise ImportError("ONNX Runtime not available. Install with: pip install onnxruntime")
    except Exception as e:
        raise RuntimeError(f"ONNX loading failed: {e}")

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
    
    # Check if it's the new format with features and labels arrays
    if 'features' in mfcc_data and 'labels' in mfcc_data:
        features_list = mfcc_data['features']
        labels = np.array(mfcc_data['labels'])
        
        # Pad features to the same length
        max_length = max(len(f) for f in features_list)
        padded_features = []
        
        for feature in features_list:
            if len(feature) < max_length:
                # Pad with zeros
                padding_needed = max_length - len(feature)
                padded_feature = feature + [[0.0] * len(feature[0]) for _ in range(padding_needed)]
                padded_features.append(padded_feature)
            else:
                padded_features.append(feature)
        
        features = np.array(padded_features)
        
        # Use the mapping from the data if available, otherwise create from unique labels
        if 'mapping' in mfcc_data:
            mapping = mfcc_data['mapping']
        else:
            # Create mapping from unique labels and convert to integers
            unique_labels = sorted(list(set(labels)))
            mapping = unique_labels
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            labels = np.array([label_to_idx[label] for label in labels])
        
        return features, labels, mapping
    else:
        # Old format - individual records
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
    """Evaluate the ONNX model and return results."""
    y_true = []
    y_pred = []
    y_probs = []
    
    for data, target in test_loader:
        # Forward pass with ONNX model
        data_np = data.detach().cpu().numpy()
        output = model.session.run(['output'], {'input': data_np})[0]
        output = torch.from_numpy(output)
        
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
    classification_rep = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    
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
    """Plot KS test visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Kolmogorov-Smirnov Test Analysis", fontsize=16, fontweight='bold')

    y_true = results['y_true']
    y_probs = results['y_probs']
    
    if y_probs is not None and len(y_probs) > 0:
        # 1. KS Statistics Bar Chart
        ax1 = axes[0, 0]
        ks_stats = results['ks_stats']
        class_labels = [class_names[i] if class_names and i < len(class_names) else f"Class {i}" 
                       for i in range(len(ks_stats))]
        
        bars = ax1.bar(range(len(ks_stats)), ks_stats, color='skyblue', edgecolor='navy', alpha=0.7)
        ax1.set_xlabel('Class')
        ax1.set_ylabel('KS Statistic')
        ax1.set_title('KS Statistics by Class')
        ax1.set_xticks(range(len(ks_stats)))
        ax1.set_xticklabels(class_labels, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, stat) in enumerate(zip(bars, ks_stats)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{stat:.3f}', ha='center', va='bottom', fontsize=9)

        # 2. KS Test Visualization (CDFs with max difference)
        ax2 = axes[0, 1]
        for class_idx in range(y_probs.shape[1]):
            class_true = y_probs[y_true == class_idx, class_idx]
            class_all = y_probs[:, class_idx]
            
            if len(class_true) > 0:
                class_name = class_labels[class_idx]
                ks_stat = results['ks_stats'][class_idx]
                
                # Sort probabilities for CDF
                sorted_true = np.sort(class_true)
                sorted_all = np.sort(class_all)
                
                # Create CDFs
                n_true = len(sorted_true)
                n_all = len(sorted_all)
                cdf_true = np.arange(1, n_true + 1) / n_true
                cdf_all = np.arange(1, n_all + 1) / n_all
                
                # Plot CDFs
                ax2.plot(sorted_true, cdf_true, label=f'{class_name} (True)', linewidth=2)
                ax2.plot(sorted_all, cdf_all, '--', label=f'{class_name} (All)', linewidth=2)
        
        ax2.set_xlabel('Class Probability')
        ax2.set_ylabel('Cumulative Distribution Function')
        ax2.set_title('CDF Comparison (True vs All Samples)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)

        # 3. Distribution Comparison (Histograms)
        ax3 = axes[1, 0]
        for class_idx in range(min(3, y_probs.shape[1])):  # Show first 3 classes
            class_true = y_probs[y_true == class_idx, class_idx]
            class_all = y_probs[:, class_idx]
            
            if len(class_true) > 0:
                class_name = class_labels[class_idx]
                ks_stat = results['ks_stats'][class_idx]
                
                # Plot histograms
                ax3.hist(class_true, bins=30, alpha=0.6, label=f'{class_name} (True)', density=True)
                ax3.hist(class_all, bins=30, alpha=0.3, label=f'{class_name} (All)', density=True)
        
        ax3.set_xlabel('Class Probability')
        ax3.set_ylabel('Density')
        ax3.set_title('Probability Distribution Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. KS Statistics Summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create summary text
        summary_text = "KS Test Summary:\n\n"
        summary_text += f"Number of classes: {len(ks_stats)}\n"
        summary_text += f"Mean KS statistic: {np.mean(ks_stats):.3f}\n"
        summary_text += f"Max KS statistic: {np.max(ks_stats):.3f}\n"
        summary_text += f"Min KS statistic: {np.min(ks_stats):.3f}\n\n"
        
        summary_text += "Interpretation:\n"
        summary_text += "• Higher KS = Better discrimination\n"
        summary_text += "• KS > 0.2 = Good discrimination\n"
        summary_text += "• KS > 0.4 = Excellent discrimination\n"
        summary_text += "• KS < 0.1 = Poor discrimination\n\n"
        
        summary_text += "Top performing classes:\n"
        sorted_indices = np.argsort(ks_stats)[::-1]
        for i, idx in enumerate(sorted_indices[:3]):
            if ks_stats[idx] > 0:
                summary_text += f"{i+1}. {class_labels[idx]}: {ks_stats[idx]:.3f}\n"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curves(results, class_names, output_path):
    """Plot ROC curves."""
    plt.figure(figsize=(10, 8))
    
    y_true = results['y_true']
    y_probs = results['y_probs']
    
    if y_probs.shape[1] == 2:  # Binary classification
        # Binary ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
        roc_auc = results['roc_auc']
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    else:  # Multi-class classification
        # Plot ROC curve for each class
        for class_idx in range(y_probs.shape[1]):
            # Binarize the labels for this class
            y_true_binary = (y_true == class_idx).astype(int)
            if len(np.unique(y_true_binary)) > 1:  # Check if class exists in test set
                fpr, tpr, _ = roc_curve(y_true_binary, y_probs[:, class_idx])
                class_name = class_names[class_idx] if class_names else f"Class {class_idx}"
                plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc_score(y_true_binary, y_probs[:, class_idx]):.3f})')
    
    # Plot diagonal line for reference
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_metrics_table(results, class_names, output_path):
    """Create a comprehensive metrics table."""
    plt.figure(figsize=(12, 8))
    
    # Get classification report
    classification_report = results.get('classification_report', {})
    
    if not classification_report:
        plt.text(0.5, 0.5, "No classification report available", 
                ha="center", va="center", transform=plt.gca().transAxes)
        plt.title("Metrics Table")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    # Prepare data for table
    metrics_data = []
    headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    
    # Add per-class metrics
    for class_name, metrics in classification_report.items():
        if isinstance(metrics, dict) and "precision" in metrics:
            # This is a class entry
            class_display_name = class_name
            if class_names and str(class_name).isdigit():
                class_idx = int(class_name)
                if class_idx < len(class_names):
                    class_display_name = class_names[class_idx]
            
            metrics_data.append([
                class_display_name,
                f"{metrics['precision']:.3f}",
                f"{metrics['recall']:.3f}",
                f"{metrics['f1-score']:.3f}",
                f"{int(metrics['support'])}"
            ])
    
    # Add macro and weighted averages
    if "macro avg" in classification_report:
        metrics_data.append([
            "Macro Avg",
            f"{classification_report['macro avg']['precision']:.3f}",
            f"{classification_report['macro avg']['recall']:.3f}",
            f"{classification_report['macro avg']['f1-score']:.3f}",
            f"{int(classification_report['macro avg']['support'])}"
        ])
    
    if "weighted avg" in classification_report:
        metrics_data.append([
            "Weighted Avg",
            f"{classification_report['weighted avg']['precision']:.3f}",
            f"{classification_report['weighted avg']['recall']:.3f}",
            f"{classification_report['weighted avg']['f1-score']:.3f}",
            f"{int(classification_report['weighted avg']['support'])}"
        ])
    
    # Create table
    if metrics_data:
        table = plt.table(
            cellText=metrics_data,
            colLabels=headers,
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(metrics_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f1f1f2')
                else:
                    table[(i, j)].set_facecolor('white')
    
    plt.axis('off')
    plt.title("Classification Metrics", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_results(results, output_path, class_names=None):
    """Save evaluation results to file."""
    # Use provided class names or default to Class_X format
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(len(results['ks_stats']))]
    
    # Save metrics to text file
    metrics_file = os.path.join(output_path, 'evaluation_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"ROC AUC: {results['roc_auc']:.4f}\n\n")
        
        f.write("KS Statistics:\n")
        for i, ks_stat in enumerate(results['ks_stats']):
            f.write(f"{class_names[i]}: {ks_stat:.4f}\n")
        
        f.write("\nClassification Report:\n")
        f.write(classification_report(results['y_true'], results['y_pred'], 
                                   target_names=class_names, zero_division=0))
    
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
    
    print("Loading ONNX model...")
    session, model_type = load_model(model_path)
    print(f"Using ONNX {model_type} model for evaluation")
    model = create_onnx_wrapper(session, model_type)
    
    print("Loading data...")
    features, labels, mapping = load_mfcc_data(data_path)
    
    print("Preprocessing features...")
    # Determine preprocessing based on ONNX model type
    if model.model_type == 'CNN':
        features = preprocess_features(features, flatten_for_rnn=False, is_cnn=True)
    elif model.model_type == 'RNN':
        features = preprocess_features(features, flatten_for_rnn=False, is_cnn=False)
    else:  # FC model
        features = preprocess_features(features, flatten_for_rnn=True, is_cnn=False)
    
    print("Creating dataset...")
    dataset = SimpleDataset(features, labels)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Use mapping from data if available (supports GTZAN, FMA, etc.)
    class_names = mapping if mapping else [f"Class_{i}" for i in range(int(labels.max()) + 1)]
    
    print("Evaluating model...")
    results = evaluate_model(model, test_loader, class_names)
    
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"ROC AUC: {results['roc_auc']:.4f}")
    
    print("Generating plots...")
    plot_confusion_matrix(results['confusion_matrix'], class_names, 
                         os.path.join(output_dir, 'confusion_matrix.png'))
    plot_roc_curves(results, class_names, os.path.join(output_dir, 'roc_curves.png'))
    plot_ks_curves(results, class_names, os.path.join(output_dir, 'ks_curves.png'))
    create_metrics_table(results, class_names, os.path.join(output_dir, 'metrics_table.png'))
    
    print("Saving results...")
    save_results(results, output_dir, class_names)
    
    print("Evaluation completed successfully!")

if __name__ == "__main__":
    main() 