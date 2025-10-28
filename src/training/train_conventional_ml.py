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


def load_mfcc_json(json_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
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
    parser = argparse.ArgumentParser(description="Train an SVM on MFCC JSON")
    parser.add_argument("--data", required=True, help="Path to MFCC JSON file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--kernel", default="rbf", choices=["linear", "rbf"], help="SVM kernel")
    parser.add_argument("--C", type=float, default=10.0, help="Regularization parameter C")
    parser.add_argument(
        "--gamma", default="scale", help="Gamma for RBF ('scale', 'auto', or float)"
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size")
    parser.add_argument(
        "--val-size", type=float, default=0.15, help="Validation split size (from train)"
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--class-weight",
        default="none",
        choices=["none", "balanced"],
        help="Handle class imbalance",
    )
    parser.add_argument("--pca", type=int, default=0, help="PCA components (0 disables)")
    parser.add_argument("--max-iter", type=int, default=1000, help="Max iterations for LinearSVC")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading: {args.data}")
    X, y, mapping = load_mfcc_json(args.data)
    print(f"Loaded features: {X.shape}, classes: {len(mapping)}")

    # Flatten time×features for SVM
    X = flatten_features(X)

    # Split train/val/test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    val_ratio = args.val_size / (1.0 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=args.random_state, stratify=y_temp
    )

    # Build pipeline: Standardize -> (optional PCA) -> SVM
    steps = [("scaler", StandardScaler(with_mean=True, with_std=True))]
    if args.pca and args.pca > 0:
        steps.append(
            ("pca", PCA(n_components=args.pca, svd_solver="auto", random_state=args.random_state))
        )

    if args.kernel == "linear":
        svm = LinearSVC(
            C=args.C,
            random_state=args.random_state,
            class_weight=None if args.class_weight == "none" else args.class_weight,
            max_iter=args.max_iter,
        )
    else:
        gamma_value = args.gamma
        if gamma_value not in {"scale", "auto"}:
            try:
                gamma_value = float(gamma_value)
            except ValueError:
                raise ValueError("--gamma must be 'scale', 'auto', or a float")
        svm = SVC(
            kernel="rbf",
            C=args.C,
            gamma=gamma_value,
            probability=False,
            random_state=args.random_state,
            class_weight=None if args.class_weight == "none" else args.class_weight,
            decision_function_shape="ovr",
        )

    steps.append(("svm", svm))
    clf = Pipeline(steps)

    print("Training SVM...")
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
            "kernel": args.kernel,
            "C": args.C,
            "gamma": args.gamma,
            "test_size": args.test_size,
            "val_size": args.val_size,
            "random_state": args.random_state,
            "class_weight": args.class_weight,
            "pca": args.pca,
            "max_iter": args.max_iter,
        },
    }

    # Save artifacts
    model_path = Path(args.output) / "svm.joblib"
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
        plt.title("SVM Confusion Matrix (Test)")
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
