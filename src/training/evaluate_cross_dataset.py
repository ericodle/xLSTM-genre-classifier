#!/usr/bin/env python3
"""
Cross-dataset evaluation for ONNX models trained on FMA:
- Evaluate on the FMA test set (primary)
- Evaluate on the GTZAN test set (secondary), aligning labels to the training mapping

Usage:
  python src/training/evaluate_cross_dataset.py \
    --model outputs/fma-run/best_model.onnx \
    --train_data mfccs/fma_13.json \
    --eval_primary mfccs/fma_13.json \
    --eval_secondary mfccs/gtzan_13.json \
    --out outputs/cross_eval/fma_model
"""

import argparse
import json
import os
from pathlib import Path
from typing import Tuple, List, Dict, Any
import re

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_onnx(model_path: str):
    import onnxruntime as ort
    return ort.InferenceSession(model_path)


def detect_onnx_model_type(session) -> str:
    try:
        input_shape = session.get_inputs()[0].shape
        if len(input_shape) == 4:
            return "CNN"
        elif len(input_shape) == 3:
            return "RNN"
        elif len(input_shape) == 2:
            return "FC"
        else:
            return "FC"
    except Exception:
        return "FC"


def load_mfcc_data(json_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    with open(json_path, 'r') as f:
        mfcc_data = json.load(f)

    if 'features' in mfcc_data and 'labels' in mfcc_data:
        features_list = mfcc_data['features']
        labels = np.array(mfcc_data['labels'])

        max_length = max(len(f) for f in features_list)
        padded_features = []
        for feature in features_list:
            if len(feature) < max_length:
                padding_needed = max_length - len(feature)
                padded_feature = feature + [[0.0] * len(feature[0]) for _ in range(padding_needed)]
                padded_features.append(padded_feature)
            else:
                padded_features.append(feature)

        features = np.array(padded_features)
        unique_labels = sorted(list(set(labels)))
        mapping = unique_labels
        if labels.dtype == object:
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            labels = np.array([label_to_idx[label] for label in labels])
        return features, labels, mapping

    # Old format
    features = []
    labels = []
    mapping: List[str] = []
    for file_path, data_dict in mfcc_data.items():
        genre = file_path.split('/')[0]
        if genre not in mapping:
            mapping.append(genre)
        genre_idx = mapping.index(genre)
        features.append(data_dict['mfcc'])
        labels.append(genre_idx)
    features = np.array(features)
    labels = np.array(labels)
    return features, labels, mapping


def preprocess_features_for_model(features: np.ndarray, model_type: str) -> np.ndarray:
    if len(features.shape) == 3:
        if model_type == 'CNN':
            return features.reshape(features.shape[0], 1, features.shape[1], features.shape[2])
        elif model_type == 'RNN':
            return features
        else:  # FC
            return features.reshape(features.shape[0], -1)
    return features


def build_dataloader(features: np.ndarray, labels: np.ndarray, batch_size: int = 32) -> DataLoader:
    ds = TensorDataset(torch.FloatTensor(features), torch.LongTensor(labels))
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def run_onnx_inference(session, x: np.ndarray) -> np.ndarray:
    return session.run(['output'], {'input': x})[0]


def evaluate(session, model_type: str, features: np.ndarray, labels: np.ndarray, class_names: List[str]) -> Dict[str, Any]:
    X = preprocess_features_for_model(features, model_type)
    loader = build_dataloader(X, labels)

    y_true = []
    y_pred = []

    for xb, yb in loader:
        xb_np = xb.detach().cpu().numpy()
        out = run_onnx_inference(session, xb_np)
        pred = np.argmax(out, axis=1)
        y_true.append(yb.numpy())
        y_pred.append(pred)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    acc = accuracy_score(y_true, y_pred)
    # Restrict report to observed label indices and map names accordingly
    observed = np.unique(y_true)
    observed = observed.astype(int).tolist()
    target_names = [str(class_names[i]) for i in observed]
    report = classification_report(
        y_true,
        y_pred,
        labels=observed,
        target_names=target_names,
        output_dict=False,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=observed).tolist()

    return {"accuracy": float(acc), "classification_report": report, "confusion_matrix": cm}


def _normalize_label(name: str) -> str:
    s = name.lower().strip()
    # Common delimiters to spaces
    s = s.replace('&', 'and').replace('/', ' ').replace('-', ' ').replace('_', ' ')
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    # Known aliases
    aliases = {
        'hip hop': 'hiphop',
        'hiphop': 'hiphop',
        'hip-hop': 'hiphop',
        'rnb': 'soulrnb',
        'soul rnb': 'soulrnb',
        'soul-and-rnb': 'soulrnb',
        'old time historic': 'oldtimehistoric',
    }
    if s in aliases:
        return aliases[s]
    # Remove remaining non-alphanumerics
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def align_secondary_dataset(features: np.ndarray, labels: np.ndarray, mapping_secondary: List[str], mapping_primary: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    # Build normalized index for primary mapping
    primary_norm = [_normalize_label(lbl) for lbl in mapping_primary]
    primary_index: Dict[str, int] = {label: i for i, label in enumerate(primary_norm)}
    keep_indices = []
    new_labels = []
    for i, lab_idx in enumerate(labels):
        label_name = mapping_secondary[lab_idx]
        norm = _normalize_label(label_name)
        if norm in primary_index:
            keep_indices.append(i)
            new_labels.append(primary_index[norm])

    if not keep_indices:
        return np.empty((0, *features.shape[1:])), np.array([], dtype=int), mapping_primary

    kept_features = features[keep_indices]
    new_labels_arr = np.array(new_labels, dtype=int)
    return kept_features, new_labels_arr, mapping_primary


def plot_confusion_matrix(cm: List[List[int]], class_names: List[str], output_path: str, title: str) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(np.array(cm), annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-dataset evaluation for ONNX models")
    parser.add_argument('--model', required=True, help='Path to ONNX model trained on FMA')
    parser.add_argument('--train_data', required=True, help='Path to training dataset JSON used for model (to get mapping)')
    parser.add_argument('--eval_primary', required=True, help='Primary eval JSON (e.g., FMA)')
    parser.add_argument('--eval_secondary', required=True, help='Secondary eval JSON (e.g., GTZAN)')
    parser.add_argument('--out', required=True, help='Output directory for results')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Load model
    session = load_onnx(args.model)
    model_type = detect_onnx_model_type(session)

    # Load mappings
    _, _, mapping_primary = load_mfcc_data(args.train_data)

    # Evaluate on primary dataset (align names if needed)
    f_feat, f_lab, f_map = load_mfcc_data(args.eval_primary)
    # Always normalize/align to training mapping to avoid name drift
    f_feat, f_lab, _ = align_secondary_dataset(f_feat, f_lab, f_map, mapping_primary)
    f_class_names = mapping_primary

    primary_results = evaluate(session, model_type, f_feat, f_lab, f_class_names)
    with open(Path(args.out) / 'evaluation_primary.json', 'w') as f:
        json.dump(primary_results, f, indent=2)
    plot_confusion_matrix(primary_results['confusion_matrix'], f_class_names, str(Path(args.out) / 'confusion_matrix_primary.png'), 'Primary Confusion Matrix')

    # Evaluate on secondary dataset (align labels to training mapping)
    s_feat, s_lab, s_map = load_mfcc_data(args.eval_secondary)
    s_feat_aligned, s_lab_aligned, s_class_names = align_secondary_dataset(s_feat, s_lab, s_map, mapping_primary)
    if s_feat_aligned.shape[0] == 0:
        secondary_results = {"accuracy": 0.0, "classification_report": "No overlapping classes; nothing to evaluate.", "confusion_matrix": []}
    else:
        secondary_results = evaluate(session, model_type, s_feat_aligned, s_lab_aligned, s_class_names)
    with open(Path(args.out) / 'evaluation_secondary.json', 'w') as f:
        json.dump(secondary_results, f, indent=2)
    plot_confusion_matrix(secondary_results['confusion_matrix'], s_class_names, str(Path(args.out) / 'confusion_matrix_secondary.png'), 'Secondary Confusion Matrix (Aligned)')

    print(f"Saved cross-eval results to: {args.out}")
    print(f"Primary (FMA) accuracy: {primary_results['accuracy']:.4f}")
    print(f"Secondary (GTZAN aligned) accuracy: {secondary_results['accuracy']:.4f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


