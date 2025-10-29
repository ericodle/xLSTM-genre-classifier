#!/usr/bin/env python3
"""
Filter overfitting analysis to only include best performing models.

This script:
1. Identifies the best performing model for each model type and dataset combination
2. Filters the overfitting analysis to only include those runs
3. Creates clean tables with only the best models
"""

import numpy as np
import pandas as pd

# Handle both direct execution and module import
try:
    from .utils import (
        AnalysisLogger,
        ensure_output_directory,
        get_model_display_name,
        get_model_order,
        save_dataframe,
    )
except ImportError:
    # For direct execution, add the parent directory to path
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from analysis.utils import (
        AnalysisLogger,
        ensure_output_directory,
        get_model_display_name,
        get_model_order,
        save_dataframe,
    )

# Initialize logger
logger = AnalysisLogger("filter_best_models")


def get_best_models():
    """Identify the best performing model for each model type and dataset."""

    # Read the results summary to get test accuracies
    results_df = pd.read_csv("outputs/analysis/results_summary.csv")

    # Remove rows with missing test_acc
    results_df = results_df.dropna(subset=["test_acc"])

    # Group by model and dataset, then find the one with highest test accuracy
    best_models = results_df.loc[results_df.groupby(["model", "dataset"])["test_acc"].idxmax()]

    # Create a mapping of (model, dataset) -> run_dir
    best_model_mapping = {}
    for _, row in best_models.iterrows():
        key = (row["model"], row["dataset"])
        best_model_mapping[key] = row["run_dir"]

    return best_model_mapping, best_models


def filter_overfitting_analysis():
    """Filter overfitting analysis to only include best performing models."""

    # Get best models mapping
    best_model_mapping, best_models_df = get_best_models()

    # Read the overfitting analysis
    overfitting_df = pd.read_csv("outputs/analysis/overfitting_analysis.csv")

    # Create a key for matching
    overfitting_df["key"] = list(zip(overfitting_df["model"], overfitting_df["dataset"]))

    # Filter to only include best models
    best_overfitting = overfitting_df[overfitting_df["key"].isin(best_model_mapping.keys())].copy()

    # Remove the key column
    best_overfitting = best_overfitting.drop("key", axis=1)

    # Remove duplicates (in case there are multiple entries for the same model-dataset combo)
    best_overfitting = best_overfitting.drop_duplicates(subset=["model", "dataset"])

    # Split by dataset
    fma_data = best_overfitting[best_overfitting["dataset"] == "FMA"].copy()
    gtzan_data = best_overfitting[best_overfitting["dataset"] == "GTZAN"].copy()

    # Format percentages
    fma_data["Training Acc (%)"] = (fma_data["train_acc"] * 100).round(1)
    fma_data["Validation Acc (%)"] = (fma_data["val_acc"] * 100).round(1)
    fma_data["Overfitting Gap (%)"] = (fma_data["overfitting"] * 100).round(1)

    gtzan_data["Training Acc (%)"] = (gtzan_data["train_acc"] * 100).round(1)
    gtzan_data["Validation Acc (%)"] = (gtzan_data["val_acc"] * 100).round(1)
    gtzan_data["Overfitting Gap (%)"] = (gtzan_data["overfitting"] * 100).round(1)

    # Sort by overfitting gap (ascending)
    fma_sorted = fma_data.sort_values("overfitting", ascending=True).reset_index(drop=True)
    gtzan_sorted = gtzan_data.sort_values("overfitting", ascending=True).reset_index(drop=True)

    # Add rank column
    fma_sorted["Rank"] = range(1, len(fma_sorted) + 1)
    gtzan_sorted["Rank"] = range(1, len(gtzan_sorted) + 1)

    # Select final columns
    fma_final = fma_sorted[
        ["Rank", "model", "Training Acc (%)", "Validation Acc (%)", "Overfitting Gap (%)"]
    ]
    gtzan_final = gtzan_sorted[
        ["Rank", "model", "Training Acc (%)", "Validation Acc (%)", "Overfitting Gap (%)"]
    ]

    # Rename columns
    fma_final.columns = [
        "Rank",
        "Model",
        "Training Acc (%)",
        "Validation Acc (%)",
        "Overfitting Gap (%)",
    ]
    gtzan_final.columns = [
        "Rank",
        "Model",
        "Training Acc (%)",
        "Validation Acc (%)",
        "Overfitting Gap (%)",
    ]

    return fma_final, gtzan_final, best_models_df


def print_table(title, df):
    """Print a formatted table."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")

    # Print header
    header = f"{'Rank':<4} {'Model':<12} {'Training Acc %':<15} {'Validation Acc %':<15} {'Overfitting %':<12}"
    print(header)
    print("-" * 80)

    # Print data rows
    for _, row in df.iterrows():
        line = f"{row['Rank']:<4} {row['Model']:<12} {row['Training Acc (%)']:<15} {row['Validation Acc (%)']:<15} {row['Overfitting Gap (%)']:<12}"
        print(line)

    # Print summary statistics
    print(f"\nSummary for {title.split(' - ')[1]}:")
    print(f"  Total Models: {len(df)}")
    print(f"  Average Overfitting: {df['Overfitting Gap (%)'].mean():.1f}%")
    print(f"  Median Overfitting: {df['Overfitting Gap (%)'].median():.1f}%")
    print(f"  Max Overfitting: {df['Overfitting Gap (%)'].max():.1f}%")
    print(f"  Min Overfitting: {df['Overfitting Gap (%)'].min():.1f}%")


def print_best_models_info(best_models_df):
    """Print information about the best models selected."""
    print(f"\n{'='*80}")
    print("BEST MODELS SELECTED (Highest Test Accuracy per Model Type)")
    print(f"{'='*80}")

    for _, row in best_models_df.iterrows():
        print(f"{row['model']} on {row['dataset']}: {row['test_acc']:.3f} test accuracy")
        print(f"  Run: {row['run_dir']}")


def main():
    """Main function."""
    print("Filtering overfitting analysis to best performing models...")

    # Get best models and filter overfitting analysis
    fma_table, gtzan_table, best_models_df = filter_overfitting_analysis()

    # Print best models info
    print_best_models_info(best_models_df)

    # Print FMA table
    print_table("OVERFITTING ANALYSIS - FMA DATASET (Best Models Only)", fma_table)

    # Print GTZAN table
    print_table("OVERFITTING ANALYSIS - GTZAN DATASET (Best Models Only)", gtzan_table)

    # Save filtered tables
    ensure_output_directory("outputs/analysis")
    save_dataframe(fma_table, "outputs/analysis/overfitting_analysis_FMA_best.csv")
    save_dataframe(gtzan_table, "outputs/analysis/overfitting_analysis_GTZAN_best.csv")

    print(f"\nFiltered tables saved:")
    print(f"  - outputs/analysis/overfitting_analysis_FMA_best.csv")
    print(f"  - outputs/analysis/overfitting_analysis_GTZAN_best.csv")

    # Overall comparison
    print(f"\n{'='*80}")
    print("DATASET COMPARISON (Best Models Only)")
    print(f"{'='*80}")
    print(f"FMA Dataset:")
    print(f"  Models: {len(fma_table)}")
    print(f"  Avg Overfitting: {fma_table['Overfitting Gap (%)'].mean():.1f}%")
    print(
        f"  Range: {fma_table['Overfitting Gap (%)'].min():.1f}% to {fma_table['Overfitting Gap (%)'].max():.1f}%"
    )
    print(f"\nGTZAN Dataset:")
    print(f"  Models: {len(gtzan_table)}")
    print(f"  Avg Overfitting: {gtzan_table['Overfitting Gap (%)'].mean():.1f}%")
    print(
        f"  Range: {gtzan_table['Overfitting Gap (%)'].min():.1f}% to {gtzan_table['Overfitting Gap (%)'].max():.1f}%"
    )


if __name__ == "__main__":
    main()
