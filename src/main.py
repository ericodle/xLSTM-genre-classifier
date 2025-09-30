"""
Main entry point for GenreDiscern.
"""

import sys
import os
import argparse
import logging
import warnings
from pathlib import Path
import json
import numpy as np
import torch
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any

# Suppress sklearn warnings about undefined metrics
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics")

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.config import Config
from core.utils import setup_logging, ensure_directory
from core.constants import (
    EXIT_SUCCESS,
    EXIT_FAILURE,
    EXIT_INTERRUPT,
    EXIT_FILE_NOT_FOUND,
    EXIT_EVALUATION_FAILED,
)
from core.data_loader import DataManager
from data.mfcc_extractor import MFCCExtractor
from data.dataset_agnostic_mfcc_extractor import (
    DatasetAgnosticMFCCExtractor,
    DatasetFactory,
)
from data.datasets.gtzan import GTZANDataset
from data.datasets.fma import FMADataset
from training.trainer import ModelTrainer


def setup_cli_parser():
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="GenreDiscern - Music Genre Classification System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract MFCC features from GTZAN dataset
  python main.py extract --input /path/to/gtzan --output /path/to/output --name gtzan_features

  # Extract MFCC features from FMA dataset
  python main.py extract --input /path/to/fma --output /path/to/output --name fma_features --dataset-type fma --fma-api-key YOUR_KEY

  # Extract MFCC features with FMA tracks CSV (faster)
  python main.py extract --input /path/to/fma --output /path/to/output --name fma_features --dataset-type fma --fma-tracks-csv /path/to/tracks.csv

  # Train a model
  python main.py train --data /path/to/features.json --model LSTM --output /path/to/output

  # Run with custom config
  python main.py train --config config.json --data data.json --model CNN --output output/
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Extract command
    extract_parser = subparsers.add_parser(
        "extract", help="Extract MFCC features from audio files"
    )
    extract_parser.add_argument(
        "--input", required=True, help="Input music directory path"
    )
    extract_parser.add_argument("--output", required=True, help="Output directory path")
    extract_parser.add_argument(
        "--name", required=True, help="Output filename (without extension)"
    )
    extract_parser.add_argument(
        "--dataset-type",
        choices=["auto", "gtzan", "fma"],
        default="auto",
        help="Dataset type (auto-detected if not specified)",
    )
    extract_parser.add_argument(
        "--fma-api-key", help="FMA API key (or set FMA_API_KEY environment variable)"
    )
    extract_parser.add_argument(
        "--fma-tracks-csv",
        type=str,
        help="Path to FMA tracks CSV file for faster processing",
    )
    extract_parser.add_argument(
        "--n-mfcc",
        type=int,
        default=13,
        help="Number of MFCC coefficients to extract (default: 13)",
    )
    extract_parser.add_argument("--config", help="Path to configuration file")

    # Update help examples
    extract_parser.epilog = """
Examples:
  # Extract from GTZAN dataset (default 13 MFCCs)
  %(prog)s extract --input ./data/gtzan --output ./mfccs --name gtzan_features
  
  # Extract from FMA dataset with 20 MFCCs
  %(prog)s extract --input ./data/fma --output ./mfccs --name fma_features \\
      --dataset-type fma --fma-api-key YOUR_API_KEY --n-mfcc 20
  
  # Auto-detect dataset with custom MFCC count
  %(prog)s extract --input ./data/audio --output ./mfccs --name features --n-mfcc 16
"""

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument(
        "--model", required=True, help="Path to trained model file"
    )
    eval_parser.add_argument(
        "--data", required=True, help="Path to MFCC features JSON file"
    )
    eval_parser.add_argument(
        "--output", required=True, help="Output directory for evaluation results"
    )
    eval_parser.add_argument("--config", help="Path to configuration file")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "--data", required=True, help="Path to MFCC features JSON file"
    )
    train_parser.add_argument("--model", required=True, help="Model type to train")
    train_parser.add_argument(
        "--output", required=True, help="Output directory for training results"
    )
    train_parser.add_argument("--config", help="Path to configuration file")
    train_parser.add_argument(
        "--epochs",
        type=int,
        help="Maximum number of training epochs (early stopping will likely end training sooner)",
    )
    train_parser.add_argument("--batch-size", type=int, help="Batch size for training")

    # Global options
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument("--log-file", help="Log file path")

    return parser


def extract_features(
    input_path: str,
    output_path: str,
    name: str,
    dataset_type: Optional[str] = None,
    fma_api_key: Optional[str] = None,
    fma_tracks_csv: Optional[str] = None,
    n_mfcc: int = 13,
    config_path: Optional[str] = None,
    logger=None,
) -> Dict[str, Any]:
    """
    Extract MFCC features from audio dataset.

    Args:
        input_path: Path to input audio dataset
        output_path: Path to output directory
        name: Name for the output file
        dataset_type: Dataset type ('gtzan', 'fma', or None for auto-detection)
        fma_api_key: FMA API key (if using FMA dataset)
        fma_tracks_csv: Path to FMA tracks CSV file
        n_mfcc: Number of MFCC coefficients to extract
        config_path: Path to configuration file
        logger: Logger instance for logging

    Returns:
        Dictionary containing extraction results
    """
    try:
        # Load configuration
        if config_path:
            config = Config(config_path)
        else:
            config = Config()

        # Override n_mfcc from command line argument
        config.audio.n_mfcc = n_mfcc

        # Set FMA API key if provided
        if fma_api_key:
            os.environ["FMA_API_KEY"] = fma_api_key

        # Determine dataset type and create appropriate extractor
        if dataset_type is None or dataset_type == "auto":
            # Auto-detect dataset type
            dataset = DatasetFactory.create_dataset(
                input_path, "auto", tracks_csv=fma_tracks_csv
            )
        else:
            # Use specified dataset type
            if dataset_type.lower() == "gtzan":
                dataset = GTZANDataset(input_path)
            elif dataset_type.lower() == "fma":
                if not fma_api_key and "FMA_API_KEY" not in os.environ:
                    raise ValueError(
                        "FMA API key is required for FMA dataset. Use --fma-api-key or set FMA_API_KEY environment variable."
                    )
                dataset = FMADataset(
                    input_path, fma_api_key or os.getenv("FMA_API_KEY"), fma_tracks_csv
                )
            else:
                raise ValueError(f"Unsupported dataset type: {dataset_type}")

        # Create extractor
        # Convert config to dictionary format expected by the extractor
        config_dict = {
            "sample_rate": config.audio.sample_rate,
            "n_mfcc": config.audio.n_mfcc,
            "hop_length": config.audio.hop_length,
            "n_fft": config.audio.n_fft,
            "max_duration": config.audio.max_duration,
        }
        extractor = DatasetAgnosticMFCCExtractor(dataset, config_dict, logger)

        # Validate dataset
        if not extractor.validate_dataset():
            raise ValueError("Dataset validation failed")

        # Create output path
        output_file = os.path.join(output_path, f"{name}.json")

        # Extract MFCCs
        if logger:
            logger.info(f"Starting MFCC extraction with {n_mfcc} coefficients...")
        result = extractor.extract_mfccs(output_file)

        if logger:
            logger.info(f"MFCC extraction completed successfully!")
            logger.info(f"Output saved to: {output_file}")
            logger.info(
                f"Features extracted: {result['metadata']['total_samples']} samples"
            )
            logger.info(f"MFCC coefficients: {n_mfcc}")

        return result

    except Exception as e:
        if logger:
            logger.error(f"MFCC extraction failed: {e}")
        raise


def train_model(args, config, logger):
    """Train a model and automatically evaluate it."""
    logger.info("Starting model training...")

    try:
        trainer = ModelTrainer(config, logger)

        # Setup training
        trainer.setup_training(
            data_path=args.data, model_type=args.model, output_dir=args.output
        )

        # Override config with command line arguments
        if args.epochs:
            config.model.max_epochs = args.epochs
        if args.batch_size:
            config.model.batch_size = args.batch_size

        # Start training
        training_history = trainer.train()

        logger.info("Model training completed successfully")

        # Automatically run evaluation after training
        logger.info("Starting automatic model evaluation...")
        try:
            evaluation_results = run_automatic_evaluation(
                trainer=trainer,
                data_path=args.data,
                output_dir=args.output,
                model_type=args.model,
                logger=logger,
            )
            logger.info("Automatic evaluation completed successfully")
            return training_history, evaluation_results
        except Exception as eval_error:
            logger.warning(f"Automatic evaluation failed: {eval_error}")
            logger.warning("Training completed successfully, but evaluation failed")
            return training_history

    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise


def run_automatic_evaluation(trainer, data_path, output_dir, model_type, logger):
    """Automatically evaluate the trained model after training completes."""
    try:
        # Create evaluation output directory
        eval_output_dir = Path(output_dir) / f"{model_type.lower()}_evaluation_results"
        eval_output_dir.mkdir(exist_ok=True)

        # Load the trained ONNX model
        onnx_path = Path(output_dir) / "best_model.onnx"
        if not onnx_path.exists():
            # Fallback to model.onnx if best_model.onnx doesn't exist
            onnx_path = Path(output_dir) / "model.onnx"
            if not onnx_path.exists():
                raise FileNotFoundError(f"No ONNX model found in {output_dir}")

        logger.info("Using ONNX model for evaluation")
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path))

        # Detect model type from ONNX
        input_shape = session.get_inputs()[0].shape
        if len(input_shape) == 4:
            detected_type = "CNN"
        elif len(input_shape) == 3:
            detected_type = "RNN"
        else:
            detected_type = "FC"

        # Create ONNX wrapper
        class ONNXModelWrapper:
            def __init__(self, session, model_type):
                self.session = session
                self.model_type = model_type
                self.eval = lambda: None

            def __call__(self, x):
                if isinstance(x, torch.Tensor):
                    x = x.detach().cpu().numpy()
                return torch.from_numpy(self.session.run(["output"], {"input": x})[0])

        model = ONNXModelWrapper(session, detected_type)

        # Load MFCC data
        logger.info("Loading MFCC data for evaluation...")

        # Check if this is a CSV file
        if data_path.lower().endswith(".csv"):
            logger.info("Detected CSV format for evaluation")
            import pandas as pd

            # Load CSV data with error handling for encoding issues
            try:
                df = pd.read_csv(data_path, encoding="utf-8")
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(data_path, encoding="latin-1")
                except UnicodeDecodeError:
                    df = pd.read_csv(data_path, encoding="cp1252")

            # Extract MFCC features (columns starting with 'mfcc')
            mfcc_columns = [col for col in df.columns if col.startswith("mfcc")]
            logger.info(f"Found {len(mfcc_columns)} MFCC features: {mfcc_columns}")

            # Extract features and labels
            features = df[mfcc_columns].values.astype(np.float32)

            # Check if we have a 'label' column (integer) or 'genre' column (string)
            if "label" in df.columns:
                labels = df["label"].values
                logger.info("Using 'label' column for target values")

                # Apply the same filtering logic as in training to ensure consistency
                from collections import Counter

                label_counts = Counter(labels)
                min_samples_per_class = 2

                # Find classes with enough samples
                valid_classes = [
                    label
                    for label, count in label_counts.items()
                    if count >= min_samples_per_class
                ]

                if len(valid_classes) < len(label_counts):
                    # Filter data to keep only classes with enough samples
                    valid_mask = np.isin(labels, valid_classes)
                    features = features[valid_mask]
                    labels = labels[valid_mask]

                    # Re-encode labels to be consecutive starting from 0
                    label_mapping = {
                        old_label: new_label
                        for new_label, old_label in enumerate(valid_classes)
                    }
                    labels = np.array([label_mapping[label] for label in labels])

                    logger.info(
                        f"Filtered to {len(valid_classes)} classes with at least {min_samples_per_class} samples each"
                    )
                    logger.info(
                        f"Removed {len(label_counts) - len(valid_classes)} classes with insufficient samples"
                    )

                # Get unique labels for mapping
                unique_labels = sorted(np.unique(labels))
                mapping = [f"class_{i}" for i in range(len(unique_labels))]

            elif "genre" in df.columns:
                # Convert string genres to integer labels
                unique_genres = sorted(df["genre"].unique())
                genre_to_label = {genre: idx for idx, genre in enumerate(unique_genres)}
                labels = np.array([genre_to_label[genre] for genre in df["genre"]])

                # Apply the same filtering logic as in training
                from collections import Counter

                label_counts = Counter(labels)
                min_samples_per_class = 2

                # Find classes with enough samples
                valid_classes = [
                    label
                    for label, count in label_counts.items()
                    if count >= min_samples_per_class
                ]

                if len(valid_classes) < len(label_counts):
                    # Filter data to keep only classes with enough samples
                    valid_mask = np.isin(labels, valid_classes)
                    features = features[valid_mask]
                    labels = labels[valid_mask]

                    # Re-encode labels to be consecutive starting from 0
                    label_mapping = {
                        old_label: new_label
                        for new_label, old_label in enumerate(valid_classes)
                    }
                    labels = np.array([label_mapping[label] for label in labels])

                    logger.info(
                        f"Filtered to {len(valid_classes)} classes with at least {min_samples_per_class} samples each"
                    )
                    logger.info(
                        f"Removed {len(label_counts) - len(valid_classes)} classes with insufficient samples"
                    )

                # Get unique genres for mapping (filtered)
                valid_genres = [
                    unique_genres[i] for i in valid_classes if i < len(unique_genres)
                ]
                mapping = valid_genres

                logger.info(
                    f"Converted {len(valid_genres)} string genres to integer labels"
                )
            else:
                raise ValueError("CSV must contain either 'label' or 'genre' column")

            logger.info(f"Feature shape: {features.shape}")
            logger.info(f"Label mapping: {dict(enumerate(mapping))}")
        else:
            # JSON format
            with open(data_path, "r") as f:
                mfcc_data = json.load(f)

            # Check if data has the new format (features/labels arrays) or old format (file paths)
            if "features" in mfcc_data and "labels" in mfcc_data:
                # New format: {"features": [...], "labels": [...]}
                logger.info("Detected new data format for evaluation")

                # Handle variable-length MFCC features by padding/truncating to consistent shape
                features_list = mfcc_data["features"]
                max_frames = max(len(feature) for feature in features_list)
                min_frames = min(len(feature) for feature in features_list)

                logger.info(
                    f"MFCC features: {len(features_list)} samples, frame range: {min_frames}-{max_frames}"
                )

                # Pad all features to the same length (max_frames)
                padded_features = []
                for feature in features_list:
                    if len(feature) < max_frames:
                        # Pad with zeros
                        padding = [
                            [0.0] * len(feature[0])
                            for _ in range(max_frames - len(feature))
                        ]
                        padded_feature = feature + padding
                    else:
                        padded_feature = feature
                    padded_features.append(padded_feature)

                features = np.array(padded_features)
                labels = np.array(mfcc_data["labels"])

                # Get unique labels for mapping and convert string labels to integers
                unique_labels = sorted(list(set(labels)))
                mapping = unique_labels

                # Convert string labels to integer indices
                label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
                labels = np.array([label_to_idx[label] for label in labels])

                logger.info(f"Padded features to shape: {features.shape}")
                logger.info(f"Label mapping: {label_to_idx}")
            else:
                # Old format: {"genre/filename.wav": {"mfcc": [...]}, ...}
                logger.info("Detected old data format for evaluation")
                features = []
                labels = []
                mapping = []

                for file_path, data_dict in mfcc_data.items():
                    genre = file_path.split("/")[0]
                    if genre not in mapping:
                        mapping.append(genre)

                    mfcc_features = np.array(data_dict["mfcc"])
                    features.append(mfcc_features)
                    labels.append(mapping.index(genre))

                features = np.array(features)
                labels = np.array(labels)

        # Preprocess features based on ONNX model type
        if model.model_type == "CNN":
            # Reshape to 4D for CNN
            if len(features.shape) == 3:
                features = features.reshape(
                    features.shape[0], 1, features.shape[1], features.shape[2]
                )
        elif model.model_type == "RNN":
            # Keep 3D for RNN
            pass
        else:
            # For FC models, flatten to 2D
            if len(features.shape) == 3:
                features = features.reshape(features.shape[0], -1)

        # Normalize features
        if len(features.shape) == 3:
            for i in range(features.shape[0]):
                features[i] = (features[i] - np.mean(features[i])) / (
                    np.std(features[i]) + 1e-8
                )
        elif len(features.shape) == 4:
            for i in range(features.shape[0]):
                features[i] = (features[i] - np.mean(features[i])) / (
                    np.std(features[i]) + 1e-8
                )
        else:
            features = (features - np.mean(features, axis=0)) / (
                np.std(features, axis=0) + 1e-8
            )

        # Create dataset and dataloader
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, features, labels):
                self.features = torch.FloatTensor(features)
                self.labels = torch.LongTensor(labels)

            def __len__(self):
                return len(self.features)

            def __getitem__(self, idx):
                return self.features[idx], self.labels[idx]

        dataset = SimpleDataset(features, labels)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

        # Run evaluation
        logger.info("Running model evaluation...")
        results = evaluate_model(model, test_loader, mapping)

        # Save evaluation results
        logger.info("Saving evaluation results...")

        # Save metrics to text file
        metrics_file = eval_output_dir / "evaluation_metrics.txt"
        with open(metrics_file, "w") as f:
            f.write(f"Model Type: {model_type}\n")
            f.write(
                f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"Overall Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"ROC AUC Score: {results['roc_auc']:.4f}\n\n")

            f.write("Classification Report:\n")
            f.write(
                classification_report(
                    results["y_true"],
                    results["y_pred"],
                    target_names=mapping,
                    zero_division=0,
                )
            )

            f.write(f"\nKS Test Statistics:\n")
            for i, ks_stat in enumerate(results["ks_stats"]):
                f.write(f"{mapping[i]}: {ks_stat:.4f}\n")

        # Plot confusion matrix
        cm_file = eval_output_dir / "confusion_matrix.png"
        plot_confusion_matrix(results["confusion_matrix"], mapping, cm_file)

        # Plot KS curves
        ks_file = eval_output_dir / "ks_curves.png"
        plot_ks_curves(results["y_true"], results["y_probs"], mapping, ks_file)

        logger.info(f"Evaluation results saved to: {eval_output_dir}")
        return results

    except Exception as e:
        logger.error(f"Automatic evaluation failed: {e}")
        raise


def evaluate_model(model, test_loader, class_names):
    """Evaluate the ONNX model and return results."""
    y_true = []
    y_pred = []
    y_probs = []

    for data, target in test_loader:
        # Forward pass with ONNX model
        data_np = data.detach().cpu().numpy()
        output = model.session.run(["output"], {"input": data_np})[0]
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
    classification_rep = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # ROC AUC scores
    roc_auc = roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")

    # KS test statistics
    ks_stats = []
    for class_idx in range(y_probs.shape[1]):
        class_true = y_probs[y_true == class_idx, class_idx]
        class_all = y_probs[:, class_idx]

        if len(class_true) > 0:
            ks_stat, _ = ks_2samp(class_true, class_all)
            ks_stats.append(ks_stat)
        else:
            # Use print for now since logger might not be in scope
            print(f"Warning: No samples found for class {class_idx}")
            ks_stats.append(0.0)

    return {
        "accuracy": accuracy,
        "classification_report": classification_rep,
        "confusion_matrix": cm,
        "roc_auc": roc_auc,
        "ks_stats": ks_stats,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_probs": y_probs,
    }


def plot_confusion_matrix(cm, class_names, output_path):
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_ks_curves(y_true, y_probs, class_names, output_path):
    """Plot KS curves for each class."""
    plt.figure(figsize=(12, 8))

    for class_idx in range(y_probs.shape[1]):
        class_true = y_probs[y_true == class_idx, class_idx]
        class_all = y_probs[:, class_idx]

        if len(class_true) > 0:
            # Calculate empirical CDFs
            class_true_sorted = np.sort(class_true)
            class_all_sorted = np.sort(class_all)

            # Plot CDFs
            plt.plot(
                class_true_sorted,
                np.linspace(0, 1, len(class_true_sorted)),
                label=f"{class_names[class_idx]} (True)",
                alpha=0.7,
            )
            plt.plot(
                class_all_sorted,
                np.linspace(0, 1, len(class_all_sorted)),
                label=f"{class_names[class_idx]} (All)",
                alpha=0.7,
                linestyle="--",
            )

    plt.xlabel("Predicted Probability")
    plt.ylabel("Cumulative Distribution")
    plt.title("Kolmogorov-Smirnov Test Curves")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Main entry point."""
    parser = setup_cli_parser()
    args = parser.parse_args()

    # If no command specified, show help
    if not args.command:
        parser.print_help()
        return EXIT_SUCCESS

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level, args.log_file)

    logger.info("GenreDiscern starting...")

    try:
        # Load configuration
        config = Config(args.config) if args.config else Config()
        logger.info("Configuration loaded successfully")

        # Execute command
        if args.command == "extract":
            try:
                result = extract_features(
                    input_path=args.input,
                    output_path=args.output,
                    name=args.name,
                    dataset_type=args.dataset_type,
                    fma_api_key=args.fma_api_key,
                    fma_tracks_csv=args.fma_tracks_csv,
                    n_mfcc=args.n_mfcc,
                    config_path=args.config,
                    logger=logger,
                )
                logger.info("Feature extraction completed successfully")
                return EXIT_SUCCESS
            except Exception as e:
                logger.error(f"Feature extraction failed: {e}")
                return EXIT_FAILURE
        elif args.command == "evaluate":
            logger.info("Starting model evaluation...")

            try:
                # Load configuration
                config = Config.load_from_file(args.config) if args.config else Config()

                # Load the trained model
                model_path = Path(args.model)
                if not model_path.exists():
                    logger.error(f"Model file not found: {args.model}")
                    sys.exit(EXIT_FILE_NOT_FOUND)

                # Load model (assuming it's a saved PyTorch model or checkpoint)
                import torch

                checkpoint = torch.load(model_path, map_location="cpu")

                # Check if it's a checkpoint dictionary or direct model
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    # It's a checkpoint, need to create model and load state dict
                    from models import get_model

                    # Get model type from config or assume FC for now
                    model = get_model("FC")  # Default to FC model
                    model.load_state_dict(checkpoint["model_state_dict"])
                    logger.info("Loaded model from checkpoint")
                else:
                    # Assume it's a direct model
                    model = checkpoint
                    logger.info("Loaded direct model")

                model.eval()

                # Load data
                data_manager = DataManager(config)

                # Load the MFCC data with the actual format
                with open(args.data, "r") as f:
                    mfcc_data = json.load(f)

                # Extract features and labels from the file path structure
                features = []
                labels = []
                mapping = []

                for file_path, data_dict in mfcc_data.items():
                    # Extract genre from file path (e.g., "reggae/reggae.00044.wav" -> "reggae")
                    genre = file_path.split("/")[0]

                    if genre not in mapping:
                        mapping.append(genre)

                    genre_idx = mapping.index(genre)
                    features.append(data_dict["mfcc"])
                    labels.append(genre_idx)

                # Convert to numpy arrays
                features = np.array(features)
                labels = np.array(labels)

                # Preprocess features
                features = data_manager.preprocess_features(features)
                labels = data_manager.encode_labels(labels)

                # Create test dataset and loader
                from torch.utils.data import DataLoader
                from core.data_loader import AudioDataset

                # Use all data for evaluation (as in your original system)
                test_dataset = AudioDataset(features, labels)
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

                # Evaluate model
                from training.evaluator import ModelEvaluator

                evaluator = ModelEvaluator(model, logger=logger)

                # GTZAN genre names
                class_names = [
                    "blues",
                    "classical",
                    "country",
                    "disco",
                    "hiphop",
                    "jazz",
                    "metal",
                    "pop",
                    "reggae",
                    "rock",
                ]

                results = evaluator.evaluate_model(test_loader, class_names)

                # Generate plots and save results
                evaluator.generate_evaluation_plots(results, args.output, class_names)
                evaluator.save_evaluation_results(
                    results, os.path.join(args.output, "evaluation_results.json")
                )

                # Print summary
                logger.info(f"Evaluation completed successfully!")
                logger.info(f"Accuracy: {results['accuracy']:.4f}")
                logger.info(f"ROC AUC: {results['roc_auc']:.4f}")
                logger.info(f"Results saved to: {args.output}")

            except Exception as e:
                logger.error(f"Model evaluation failed: {e}")
                sys.exit(EXIT_EVALUATION_FAILED)
        elif args.command == "train":
            result = train_model(args, config, logger)
            if isinstance(result, tuple):
                training_history, evaluation_results = result
                logger.info("Training and evaluation completed successfully")
                logger.info(
                    f"Final training accuracy: {training_history['val_acc'][-1]:.4f}"
                )
                if evaluation_results:
                    logger.info(
                        f"Evaluation accuracy: {evaluation_results['accuracy']:.4f}"
                    )
            else:
                training_history = result
                logger.info("Training completed successfully")
                logger.info(
                    f"Final training accuracy: {training_history['val_acc'][-1]:.4f}"
                )
        else:
            logger.error(f"Unknown command: {args.command}")
            parser.print_help()
            return EXIT_FAILURE

        logger.info("GenreDiscern completed successfully")
        return EXIT_SUCCESS

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return EXIT_INTERRUPT
    except Exception as e:
        logger.error(f"GenreDiscern failed: {e}")
        return EXIT_FAILURE


if __name__ == "__main__":
    main()
