#!/usr/bin/env python3
"""
Aggregate and visualize comparative results across runs.

- Scans ./outputs/* recursively for:
  - results.json (training runs)
  - *_training_metadata.json (for ONNX runs)
  - SVM results.json
- Extracts dataset (gtzan/fma), model type, and key metrics.
- Saves summary CSV and comparative plots.

Usage:
  python analyze_results.py --outputs ./outputs --out ./outputs/analysis
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set a clean theme with larger fonts
sns.set_theme(style="whitegrid", context="talk")

# Display label mapping for models in plots
MODEL_LABEL_MAP = {
    "TRANSFORMER": "TR",
}


def infer_dataset_from_path(p: str) -> str:
    s = p.lower()
    if "gtzan" in s:
        return "GTZAN"
    if "fma" in s:
        return "FMA"
    return "UNKNOWN"


def infer_model_from_path_or_json(dir_path: Path, payload: Dict[str, Any]) -> str:
    # Try directory naming first (handles patterns like cnn-*, lstm-*, xlstm-*, tr-*)
    name = dir_path.name.lower()
    # Explicit short alias for Transformer
    if name.startswith("tr-") or "-tr-" in name or name.endswith("-tr"):
        return "TRANSFORMER"
    # Order matters: check 'xlstm' before 'lstm' to avoid substring collisions
    for m in ["xlstm", "transformer", "vgg", "cnn", "lstm", "gru", "svm", "fc"]:
        if m in name:
            return m.upper()
    # Try JSON fields
    # SVM script stores params but not model string, tag as SVM if present
    if "params" in payload and "kernel" in payload["params"]:
        return "SVM"
    return payload.get("model_type", "UNKNOWN").upper()


def collect_results(outputs_dir: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for root, dirs, files in os.walk(outputs_dir):
        root_path = Path(root)
        if "results.json" in files:
            try:
                with open(root_path / "results.json", "r") as f:
                    data = json.load(f)
            except Exception:
                continue

            dataset = infer_dataset_from_path(str(root_path))
            model = infer_model_from_path_or_json(root_path, data)

            # SVM format
            if all(k in data for k in ["train", "val", "test"]):
                def acc_of(split: str) -> float:
                    try:
                        return float(data[split]["accuracy"])  # already float
                    except Exception:
                        return float("nan")

                rows.append(
                    {
                        "run_dir": str(root_path),
                        "dataset": dataset,
                        "model": model,
                        "train_acc": acc_of("train"),
                        "val_acc": acc_of("val"),
                        "test_acc": acc_of("test"),
                    }
                )
                continue

        # Neural network evaluation: parse evaluation/evaluation_metrics.txt
        eval_dir = root_path / "evaluation"
        eval_file = eval_dir / "evaluation_metrics.txt"
        if eval_file.exists():
            try:
                with open(eval_file, "r") as f:
                    text = f.read()
                # Expect lines like: Accuracy: 0.8123, ROC AUC: 0.91...
                import re
                acc_match = re.search(r"Accuracy:\s*([0-9]*\.?[0-9]+)", text)
                roc_match = re.search(r"ROC AUC:\s*([0-9]*\.?[0-9]+)", text)
                test_acc: Optional[float] = float(acc_match.group(1)) if acc_match else np.nan
                roc_auc: Optional[float] = float(roc_match.group(1)) if roc_match else np.nan
            except Exception:
                test_acc = np.nan
                roc_auc = np.nan

            dataset = infer_dataset_from_path(str(root_path))
            model = infer_model_from_path_or_json(root_path, {})
            rows.append(
                {
                    "run_dir": str(root_path),
                    "dataset": dataset,
                    "model": model,
                    "train_acc": np.nan,
                    "val_acc": np.nan,
                    "test_acc": test_acc,
                    "roc_auc": roc_auc,
                }
            )
            continue

        # Neural network evaluation: parse evaluation/evaluation_results.json (preferred)
        eval_json = eval_dir / "evaluation_results.json"
        if eval_json.exists():
            try:
                with open(eval_json, "r") as f:
                    payload = json.load(f)
                test_acc = float(payload.get("accuracy", np.nan))
                roc_auc = float(payload.get("roc_auc", np.nan))
            except Exception:
                test_acc = np.nan
                roc_auc = np.nan

            dataset = infer_dataset_from_path(str(root_path))
            model = infer_model_from_path_or_json(root_path, payload)
            rows.append(
                {
                    "run_dir": str(root_path),
                    "dataset": dataset,
                    "model": model,
                    "train_acc": np.nan,
                    "val_acc": np.nan,
                    "test_acc": test_acc,
                    "roc_auc": roc_auc,
                }
            )
            continue

        # ONNX/e2e training metadata (optional)
        for f in files:
            if f.endswith("_training_metadata.json"):
                try:
                    with open(root_path / f, "r") as fp:
                        meta = json.load(fp)
                except Exception:
                    continue
                dataset = infer_dataset_from_path(str(root_path))
                model = infer_model_from_path_or_json(root_path, meta)
                rows.append(
                    {
                        "run_dir": str(root_path),
                        "dataset": dataset,
                        "model": model,
                        "train_acc": np.nan,
                        "val_acc": np.nan,
                        "test_acc": np.nan,
                        "roc_auc": np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _annotate_bars(ax: plt.Axes, fmt: str = "{:.2f}") -> None:
    for p in ax.patches:
        height = p.get_height()
        if np.isnan(height) or height <= 1e-6:
            continue
        ax.annotate(
            fmt.format(height),
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=10,
            xytext=(0, 3),
            textcoords="offset points",
        )


def plot_bars(df: pd.DataFrame, out_dir: Path, metric: str) -> None:
    if df.empty:
        return
    plt.figure(figsize=(12, 7))
    df_plot = df.copy()
    df_plot["model"] = df_plot["model"].replace(MODEL_LABEL_MAP)
    # Drop rows with missing metric to avoid zero bars/labels
    df_plot = df_plot[~df_plot[metric].isna()]
    if df_plot.empty:
        plt.close()
        return
    ax = sns.barplot(
        data=df_plot,
        x="model",
        y=metric,
        hue="dataset",
        errorbar=None,
    )
    ax.set_title(f"{metric} by model and dataset")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Model")
    ax.set_ylabel(metric.replace("_", " ").title())
    # Rotate labels for readability
    for label in ax.get_xticklabels():
        label.set_rotation(20)
        label.set_horizontalalignment("right")
    # Place legend outside if crowded
    ax.legend(title="Dataset", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    # Optional: annotate bars
    _annotate_bars(ax)
    plt.tight_layout()
    path = out_dir / f"{metric}_by_model_dataset.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_model_grid(df: pd.DataFrame, out_dir: Path) -> None:
    # Faceted bar plots per dataset
    if df.empty:
        return
    for metric in ["train_acc", "val_acc", "test_acc"]:
        df_plot = df.copy()
        df_plot["model"] = df_plot["model"].replace(MODEL_LABEL_MAP)
        df_plot = df_plot[~df_plot[metric].isna()]
        if df_plot.empty:
            continue
        g = sns.catplot(
            data=df_plot,
            x="model",
            y=metric,
            col="dataset",
            kind="bar",
            errorbar=None,
            height=4.5,
            aspect=1.1,
        )
        g.set(ylim=(0, 1))
        g.set_xlabels("Model")
        g.set_ylabels(metric.replace("_", " ").title())
        # Rotate tick labels
        for ax in g.axes.flatten():
            for label in ax.get_xticklabels():
                label.set_rotation(20)
                label.set_horizontalalignment("right")
        g.fig.suptitle(f"{metric} by model per dataset", y=1.05)
        path = out_dir / f"{metric}_by_model_per_dataset.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate and plot results from outputs/")
    parser.add_argument("--outputs", default="./outputs", help="Outputs root directory")
    parser.add_argument("--out", default="./outputs/analysis", help="Where to save analysis")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = collect_results(args.outputs)
    if df.empty:
        print("No results found.")
        return 0

    # Save raw table
    csv_path = out_dir / "results_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved summary: {csv_path}")

    # Aggregate means per (dataset, model)
    agg = (
        df.groupby(["dataset", "model"], dropna=False)[["train_acc", "val_acc", "test_acc"]]
        .mean(numeric_only=True)
        .reset_index()
    )
    agg_csv = out_dir / "results_agg.csv"
    agg.to_csv(agg_csv, index=False)
    print(f"Saved aggregates: {agg_csv}")

    # Plots
    plot_bars(agg, out_dir, "test_acc")
    plot_bars(agg, out_dir, "val_acc")
    plot_bars(agg, out_dir, "train_acc")
    plot_model_grid(df, out_dir)

    print(f"Plots saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
