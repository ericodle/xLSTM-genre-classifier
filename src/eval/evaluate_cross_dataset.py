import json
import os
import sys
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

# Suppress sklearn warnings about undefined metrics
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics")

# Add src directory to path
sys.path.insert(0, "src")

from eval.data_utils import DataLoaderUtils
from eval.model_loader import UnifiedModelLoader
from eval.plotting_utils import PlottingUtilities


def load_model(model_path):
    """Load a trained model from ONNX or joblib file."""
    loader = UnifiedModelLoader()
    return loader.load_model(model_path)


def load_mfcc_data(json_path):
    """Load MFCC data from the JSON file."""
    data_utils = DataLoaderUtils()
    return data_utils.load_mfcc_data(json_path)


def preprocess_features(features, model_type=None, flatten_for_rnn=False, is_cnn=False):
    """Preprocess features (normalization, reshaping, etc.)."""
    # Determine model type from parameters
    if model_type is None:
        if is_cnn:
            model_type = "CNN"
        elif flatten_for_rnn:
            model_type = "FC"
        else:
            model_type = "RNN"

    data_utils = DataLoaderUtils()
    return data_utils.preprocess_features(features, model_type)


class SimpleDataset(torch.utils.data.Dataset):
    """Simple dataset for evaluation."""

    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def evaluate_model(model, dataloader, class_names):
    """Evaluate the model on the given dataloader."""
    y_true = []
    y_pred = []
    y_probs = []

    model.eval()
    with torch.no_grad():
        for data, target in dataloader:
            # Forward pass
            output = model(data)

            # Get probabilities and predictions
            if hasattr(output, "log_softmax"):
                # Handle log_softmax output
                probs = torch.exp(output)
            else:
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
    from scipy.stats import ks_2samp
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

    accuracy = (y_true == y_pred).mean()

    # Classification report
    if class_names:
        report = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
        )
    else:
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # ROC AUC scores
    if y_probs.shape[1] == 2:  # Binary classification
        roc_auc = roc_auc_score(y_true, y_probs[:, 1])
    else:  # Multi-class classification
        # Ensure probabilities sum to 1.0 for each sample
        y_probs_normalized = y_probs / y_probs.sum(axis=1, keepdims=True)
        try:
            roc_auc = roc_auc_score(y_true, y_probs_normalized, multi_class="ovr", average="macro")
        except ValueError as e:
            print(f"Warning: ROC AUC calculation failed: {e}")
            roc_auc = 0.0

    # KS test statistics
    ks_test_stats = []
    for class_idx in range(y_probs.shape[1]):
        # Get probabilities for samples that actually belong to this class
        class_true = y_probs[y_true == class_idx, class_idx]
        # Get probabilities for all samples for this class
        class_all = y_probs[:, class_idx]

        if len(class_true) > 0:
            ks_stat, _ = ks_2samp(class_true, class_all)
            ks_test_stats.append(ks_stat)
        else:
            ks_test_stats.append(0.0)

    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm,
        "roc_auc": roc_auc,
        "ks_stats": ks_test_stats,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_probs": y_probs,
    }


def plot_confusion_matrix(cm, class_names, output_path):
    """Plot confusion matrix."""
    plotting = PlottingUtilities()
    plotting.plot_confusion_matrix(cm, class_names, output_path)


def plot_ks_curves(results, class_names, output_path):
    """Plot KS test visualizations."""
    plotting = PlottingUtilities()
    plotting.plot_ks_curves(results, class_names, output_path)


def plot_roc_curves(results, class_names, output_path):
    """Plot ROC curves."""
    plotting = PlottingUtilities()
    plotting.plot_roc_curves(results, class_names, output_path)


def create_metrics_table(results, class_names, output_path):
    """Create a comprehensive metrics table."""
    plotting = PlottingUtilities()
    plotting.create_metrics_table(results, class_names, output_path)


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Cross-dataset evaluation")
    parser.add_argument("--model", required=True, help="Path to the ONNX model file")
    parser.add_argument("--train-data", required=True, help="Path to the training data JSON file")
    parser.add_argument(
        "--eval-primary", required=True, help="Path to the primary evaluation data JSON file"
    )
    parser.add_argument(
        "--eval-secondary", required=True, help="Path to the secondary evaluation data JSON file"
    )
    parser.add_argument("--out", required=True, help="Output directory for results")
    parser.add_argument("--model-type", help="Model type (auto-detected if not specified)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.out, exist_ok=True)

    # Load model
    print(f"Loading model from {args.model}...")
    model = load_model(args.model)

    # Get model type from loaded model
    model_type = getattr(model, "model_type", args.model_type)
    if model_type is None:
        raise ValueError("Could not determine model type. Please specify --model-type")

    # Load training data to get the mapping
    print(f"Loading training data from {args.train_data}...")
    _, _, train_mapping = load_mfcc_data(args.train_data)

    # Load primary evaluation data
    print(f"Loading primary evaluation data from {args.eval_primary}...")
    features_primary, labels_primary, primary_mapping = load_mfcc_data(args.eval_primary)

    # Load secondary evaluation data
    print(f"Loading secondary evaluation data from {args.eval_secondary}...")
    features_secondary, labels_secondary, secondary_mapping = load_mfcc_data(args.eval_secondary)

    # Preprocess features based on model type
    print("Preprocessing features...")
    features_primary = preprocess_features(features_primary, model_type=model_type)
    features_secondary = preprocess_features(features_secondary, model_type=model_type)

    # Create datasets and dataloaders
    dataset_primary = SimpleDataset(features_primary, labels_primary)
    dataloader_primary = DataLoader(dataset_primary, batch_size=args.batch_size, shuffle=False)

    dataset_secondary = SimpleDataset(features_secondary, labels_secondary)
    dataloader_secondary = DataLoader(dataset_secondary, batch_size=args.batch_size, shuffle=False)

    # Evaluate on primary dataset
    print("Evaluating on primary dataset...")
    results_primary = evaluate_model(model, dataloader_primary, train_mapping)

    # Evaluate on secondary dataset
    print("Evaluating on secondary dataset...")
    results_secondary = evaluate_model(model, dataloader_secondary, train_mapping)

    # Generate plots for primary dataset
    print("Generating plots for primary dataset...")
    plot_confusion_matrix(
        results_primary["confusion_matrix"],
        train_mapping,
        os.path.join(args.out, "primary_confusion_matrix.png"),
    )
    plot_ks_curves(results_primary, train_mapping, os.path.join(args.out, "primary_ks_curves.png"))
    plot_roc_curves(
        results_primary, train_mapping, os.path.join(args.out, "primary_roc_curves.png")
    )
    create_metrics_table(
        results_primary, train_mapping, os.path.join(args.out, "primary_metrics_table.png")
    )

    # Generate plots for secondary dataset
    print("Generating plots for secondary dataset...")
    plot_confusion_matrix(
        results_secondary["confusion_matrix"],
        train_mapping,
        os.path.join(args.out, "secondary_confusion_matrix.png"),
    )
    plot_ks_curves(
        results_secondary, train_mapping, os.path.join(args.out, "secondary_ks_curves.png")
    )
    plot_roc_curves(
        results_secondary, train_mapping, os.path.join(args.out, "secondary_roc_curves.png")
    )
    create_metrics_table(
        results_secondary, train_mapping, os.path.join(args.out, "secondary_metrics_table.png")
    )

    # Save results
    results_path = os.path.join(args.out, "cross_evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump({"primary": results_primary, "secondary": results_secondary}, f, indent=2)

    print(f"Cross-dataset evaluation complete! Results saved to {args.out}")
    print(f"Primary dataset accuracy: {results_primary['accuracy']:.4f}")
    print(f"Secondary dataset accuracy: {results_secondary['accuracy']:.4f}")


if __name__ == "__main__":
    main()
