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
  python analyze_results.py --input-dir ./outputs --output-dir ./outputs/analysis
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Handle both direct execution and module import
try:
    from .utils import (
        AnalysisLogger,
        ensure_output_directory,
        get_dataset_colors,
        get_model_display_name,
        get_model_order,
        infer_dataset_from_path,
        infer_model_from_path,
        load_json_data,
        save_dataframe,
        save_plot,
        setup_plotting_style,
    )
except ImportError:
    # For direct execution, add the parent directory to path
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from analysis.utils import (
        AnalysisLogger,
        ensure_output_directory,
        get_dataset_colors,
        get_model_display_name,
        get_model_order,
        infer_dataset_from_path,
        infer_model_from_path,
        load_json_data,
        save_dataframe,
        save_plot,
        setup_plotting_style,
    )

# Initialize logger and plotting
logger = AnalysisLogger("analyze_results")
setup_plotting_style()


def collect_results(outputs_dir: str) -> pd.DataFrame:
    """Collect results from all training runs."""
    rows: List[Dict[str, Any]] = []

    if not os.path.exists(outputs_dir):
        logger.error(f"Outputs directory does not exist: {outputs_dir}")
        return pd.DataFrame()

    logger.info(f"Scanning directory: {outputs_dir}")

    for root, dirs, files in os.walk(outputs_dir):
        root_path = Path(root)
        if "results.json" in files:
            try:
                data = load_json_data(str(root_path / "results.json"))
            except Exception as e:
                logger.warning(f"Failed to load {root_path / 'results.json'}: {e}")
                continue

            dataset = infer_dataset_from_path(str(root_path))
            model = infer_model_from_path(str(root_path), data)

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
            model = infer_model_from_path(str(root_path), {})
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
            model = infer_model_from_path(str(root_path), payload)
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
                model = infer_model_from_path(str(root_path), meta)
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

    # Create figure with better proportions for conference presentation
    fig, ax = plt.subplots(figsize=(16, 8))
    df_plot = df.copy()
    df_plot["model"] = df_plot["model"].apply(get_model_display_name)
    # Drop rows with missing metric to avoid zero bars/labels
    df_plot = df_plot[~df_plot[metric].isna()]
    if df_plot.empty:
        plt.close()
        return

    # Filter to only include models in our desired order
    df_plot = df_plot[df_plot["model"].isin(get_model_order())]

    # Get unique datasets and models
    datasets = df_plot["dataset"].unique()
    models = get_model_order()

    # Set up bar positions
    x = np.arange(len(models))
    width = 0.35  # Width of bars

    # Create bars for each dataset
    bars = []
    dataset_colors = get_dataset_colors()
    for i, dataset in enumerate(datasets):
        dataset_data = df_plot[df_plot["dataset"] == dataset]
        values = []
        for model in models:
            model_data = dataset_data[dataset_data["model"] == model]
            if len(model_data) > 0:
                values.append(model_data[metric].iloc[0])
            else:
                values.append(0)

        # Offset bars for grouped display
        x_pos = x + i * width - width * (len(datasets) - 1) / 2

        bar = ax.bar(
            x_pos,
            values,
            width,
            label=dataset,
            color=dataset_colors.get(dataset, "#666666"),
            alpha=0.8,
            edgecolor="white",
            linewidth=1.5,
        )
        bars.extend(bar)

        # Add value labels on top of bars
        for j, (bar_item, value) in enumerate(zip(bar, values)):
            if value > 0:
                ax.text(
                    bar_item.get_x() + bar_item.get_width() / 2.0,
                    value + 0.01,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

    # Customize the plot
    ax.set_title(f"Test Accuracy by Model Type", fontsize=18, fontweight="bold", pad=20)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Model Type", fontsize=14, fontweight="bold")
    ax.set_ylabel("Test Accuracy", fontsize=14, fontweight="bold")

    # Set x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(
        [get_model_display_name(model) for model in models], rotation=45, ha="right", fontsize=12
    )

    # Add legend in top right
    ax.legend(loc="upper right", fontsize=12, framealpha=0.9)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    # Customize spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    # Add subtle background
    ax.set_facecolor("#FAFAFA")

    # Adjust layout and save
    plt.tight_layout()
    path = out_dir / f"{metric}_by_model_dataset.png"
    save_plot(plt.gcf(), str(path))


def plot_model_grid(df: pd.DataFrame, out_dir: Path) -> None:
    # Faceted bar plots per dataset
    if df.empty:
        return
    for metric in ["train_acc", "val_acc", "test_acc"]:
        df_plot = df.copy()
        df_plot["model"] = df_plot["model"].apply(get_model_display_name)
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
        save_plot(plt.gcf(), str(path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate and plot results from training outputs")
    parser.add_argument(
        "--input-dir",
        "--input",
        default="./outputs",
        help="Directory containing training results to analyze",
    )
    parser.add_argument(
        "--output-dir",
        "--output",
        default="./outputs/analysis",
        help="Directory where analysis results will be saved",
    )
    args = parser.parse_args()

    try:
        # Validate input directory
        if not os.path.exists(args.input_dir):
            logger.error(f"Input directory does not exist: {args.input_dir}")
            return 1

        # Create output directory
        out_dir = ensure_output_directory(args.output_dir)
        logger.info(f"Output directory: {out_dir}")

        # Collect results
        df = collect_results(args.input_dir)
        if df.empty:
            logger.warning("No results found.")
            return 0

        logger.info(f"Found {len(df)} results")

        # Save raw table
        csv_path = out_dir / "results_summary.csv"
        save_dataframe(df, str(csv_path))
        logger.info(f"Saved summary: {csv_path}")

        # Aggregate maximum (best) per (dataset, model)
        agg = (
            df.groupby(["dataset", "model"], dropna=False)[["train_acc", "val_acc", "test_acc"]]
            .max(numeric_only=True)
            .reset_index()
        )
        agg_csv = out_dir / "results_agg.csv"
        save_dataframe(agg, str(agg_csv))
        logger.info(f"Saved aggregates: {agg_csv}")

        # Generate plots
        plot_bars(agg, out_dir, "test_acc")
        logger.info("Generated plots")

        return 0

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
