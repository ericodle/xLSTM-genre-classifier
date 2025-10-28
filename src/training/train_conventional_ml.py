#!/usr/bin/env python3
"""
Train and evaluate conventional ML models (SVM, Random Forest, Naive Bayes, KNN) on MFCC JSON datasets.

Example usage:
  # SVM
  python src/training/train_conventional_ml.py --data gtzan-data/mfccs_splits/train.json --model svm --kernel rbf --C 10 --output outputs/svm-gtzan

  # Random Forest
  python src/training/train_conventional_ml.py --data gtzan-data/mfccs_splits/train.json --model rf --n-estimators 100 --output outputs/rf-gtzan

  # Naive Bayes
  python src/training/train_conventional_ml.py --data gtzan-data/mfccs_splits/train.json --model nb --output outputs/nb-gtzan

  # KNN
  python src/training/train_conventional_ml.py --data gtzan-data/mfccs_splits/train.json --model knn --n-neighbors 5 --output outputs/knn-gtzan
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.conventional_ml import get_conventional_model


def load_mfcc_json(data_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load MFCC data from either a single JSON file or pre-split directory."""
    if os.path.isdir(data_path):
        # Pre-split format: load from train.json
        train_json = os.path.join(data_path, "train.json")
        if not os.path.exists(train_json):
            raise ValueError(f"train.json not found in directory: {data_path}")
        json_path = train_json
    else:
        # Single file format
        json_path = data_path

    with open(json_path, "r") as f:
        data = json.load(f)

    # New format
    if "features" in data and "labels" in data:
        features_list = data["features"]
        labels_list = data["labels"]

        # Pad variable-length sequences to the same length
        max_len = max(len(f) for f in features_list)
        feat_dim = len(features_list[0][0]) if max_len > 0 else 0
        padded = []
        for f in features_list:
            if len(f) < max_len:
                pad = [[0.0] * feat_dim for _ in range(max_len - len(f))]
                f = f + pad
            padded.append(f)

        X = np.asarray(padded, dtype=np.float32)  # (N, T, D)
        y = np.asarray(labels_list)
        mapping = data.get("mapping", sorted(list(set(labels_list))))
        # If labels are strings, map to ints
        if y.dtype == object:
            label_to_idx = {label: i for i, label in enumerate(sorted(set(labels_list)))}
            y = np.asarray([label_to_idx[label] for label in labels_list])
            mapping = [k for k, _ in sorted(label_to_idx.items(), key=lambda kv: kv[1])]
        return X, y, mapping

    # Old format
    features = []
    labels = []
    mapping = []
    for file_path, rec in data.items():
        genre = file_path.split("/")[0]
        if genre not in mapping:
            mapping.append(genre)
        genre_idx = mapping.index(genre)
        features.append(rec["mfcc"])  # (T, D)
        labels.append(genre_idx)

    # Pad variable-length
    max_len = max(len(f) for f in features)
    feat_dim = len(features[0][0]) if max_len > 0 else 0
    padded = []
    for f in features:
        if len(f) < max_len:
            pad = [[0.0] * feat_dim for _ in range(max_len - len(f))]
            f = f + pad
        padded.append(f)

    X = np.asarray(padded, dtype=np.float32)
    y = np.asarray(labels)
    return X, y, mapping


def flatten_features(X: np.ndarray) -> np.ndarray:
    if X.ndim == 3:
        # (N, T, D) -> (N, T*D)
        return X.reshape(X.shape[0], -1)
    return X


def main() -> int:
    parser = argparse.ArgumentParser(description="Train conventional ML models on MFCC JSON")
    parser.add_argument(
        "--data", required=True, help="Path to MFCC JSON file or pre-split directory"
    )
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--model",
        required=True,
        choices=["svm", "rf", "nb", "knn"],
        help="Model type: svm, rf (Random Forest), nb (Naive Bayes), knn (K-Nearest Neighbors)",
    )

    # Common arguments
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size")
    parser.add_argument("--val-size", type=float, default=0.15, help="Validation split size")

    # SVM-specific arguments
    parser.add_argument("--kernel", default="rbf", choices=["linear", "rbf"], help="SVM kernel")
    parser.add_argument("--C", type=float, default=10.0, help="SVM regularization parameter C")
    parser.add_argument(
        "--gamma", default="scale", help="SVM gamma for RBF ('scale', 'auto', or float)"
    )

    # Random Forest-specific arguments
    parser.add_argument("--n-estimators", type=int, default=100, help="RF number of estimators")
    parser.add_argument("--max-depth", type=int, default=None, help="RF max depth")

    # KNN-specific arguments
    parser.add_argument("--n-neighbors", type=int, default=5, help="KNN number of neighbors")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading: {args.data}")

    if os.path.isdir(args.data):
        # Pre-split format: load all three splits
        train_json = os.path.join(args.data, "train.json")
        val_json = os.path.join(args.data, "val.json")
        test_json = os.path.join(args.data, "test.json")

        if not all(os.path.exists(f) for f in [train_json, val_json, test_json]):
            raise ValueError(f"Missing split files in directory: {args.data}")

        # Load all splits
        X_train, y_train, mapping = load_mfcc_json(train_json)
        X_val, y_val, _ = load_mfcc_json(val_json)
        X_test, y_test, _ = load_mfcc_json(test_json)

        print(f"Loaded pre-split data:")
        print(f"  Train: {X_train.shape[0]} samples")
        print(f"  Val:   {X_val.shape[0]} samples")
        print(f"  Test:  {X_test.shape[0]} samples")
        print(f"  Classes: {len(mapping)}")

        # Flatten time×features for conventional ML models
        X_train = flatten_features(X_train)
        X_val = flatten_features(X_val)
        X_test = flatten_features(X_test)

    else:
        # Single file format: load and split
        X, y, mapping = load_mfcc_json(args.data)
        print(f"Loaded features: {X.shape}, classes: {len(mapping)}")

        # Flatten time×features for conventional ML models
        X = flatten_features(X)

        # Split train/val/test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
        )
        val_ratio = args.val_size / (1.0 - args.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=args.random_state, stratify=y_temp
        )

    # Prepare model-specific parameters
    model_params = {}

    if args.model == "svm":
        model_params = {
            "kernel": args.kernel,
            "C": args.C,
            "gamma": args.gamma,
            "random_state": args.random_state,
        }
    elif args.model == "rf":
        model_params = {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "random_state": args.random_state,
        }
    elif args.model == "nb":
        # GaussianNB doesn't accept random_state
        model_params = {}
    elif args.model == "knn":
        # KNN doesn't accept random_state
        model_params = {
            "n_neighbors": args.n_neighbors,
        }

    # Create model using factory
    model_name = {"svm": "SVM", "rf": "Random Forest", "nb": "Naive Bayes", "knn": "KNN"}[
        args.model
    ]
    print(f"Training {model_name}...")

    clf = get_conventional_model(args.model, **model_params)
    clf.fit(X_train, y_train)

    # Evaluation
    def evaluate(split_name: str, Xs: np.ndarray, ys: np.ndarray) -> dict:
        preds = clf.predict(Xs)
        acc = accuracy_score(ys, preds)
        rep = classification_report(
            ys, preds, target_names=[str(m) for m in mapping], zero_division=0
        )
        cm = confusion_matrix(ys, preds)
        print(f"{split_name} accuracy: {acc:.4f}")
        return {"accuracy": float(acc), "report": rep, "confusion_matrix": cm.tolist()}

    results = {
        "train": evaluate("Train", X_train, y_train),
        "val": evaluate("Val", X_val, y_val),
        "test": evaluate("Test", X_test, y_test),
        "mapping": mapping,
        "params": {
            "model": args.model,
            "test_size": args.test_size,
            "val_size": args.val_size,
            "random_state": args.random_state,
            **model_params,
        },
    }

    # Save artifacts
    model_path = Path(args.output) / f"{args.model}.joblib"
    results_path = Path(args.output) / "results.json"

    # Ensure the output directory exists (defensive programming)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to absolute path to avoid any path issues
    model_path_abs = str(model_path.absolute())
    results_path_abs = str(results_path.absolute())

    print(f"Saving model to: {model_path_abs}")
    print(f"Saving results to: {results_path_abs}")

    joblib.dump(clf, model_path_abs)
    with open(results_path_abs, "w") as f:
        json.dump(results, f, indent=2)

    # Plot and save confusion matrix for test split
    try:
        cm = np.array(results["test"]["confusion_matrix"], dtype=np.int64)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[str(m) for m in mapping],
            yticklabels=[str(m) for m in mapping],
        )
        plt.title(f"{model_name} Confusion Matrix (Test)")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        cm_path = Path(args.output) / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved confusion matrix to: {cm_path}")
    except Exception as e:
        print(f"Warning: failed to save confusion matrix plot: {e}")

    print(f"Saved model to: {model_path_abs}")
    print(f"Saved results to: {results_path_abs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
