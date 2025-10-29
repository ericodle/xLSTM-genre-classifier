#!/usr/bin/env python3
"""
Comprehensive overfitting analysis tool.

This script:
1. Finds the best performing model for each model type and dataset based on test accuracy
2. Gets the training and validation accuracy from the last epoch of those best models
3. Calculates overfitting gap (train_acc - val_acc) for those best models
4. Creates a final formatted table in a specified order
5. Saves both raw analysis and formatted results
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Handle both direct execution and module import
try:
    from .utils import (
        AnalysisLogger,
        ensure_output_directory,
        get_model_display_name,
        get_model_order,
        infer_dataset_from_path,
        infer_model_from_path,
        load_json_data,
        safe_divide,
        save_dataframe,
    )
except ImportError:
    # For direct execution, add the parent directory to path
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from analysis.utils import (
        AnalysisLogger,
        ensure_output_directory,
        get_model_display_name,
        get_model_order,
        infer_dataset_from_path,
        infer_model_from_path,
        load_json_data,
        safe_divide,
        save_dataframe,
    )

# Initialize logger
logger = AnalysisLogger("overfitting_analysis")


def extract_model_info(run_dir):
    """Extract model information from directory name."""
    model = infer_model_from_path(run_dir)
    dataset = infer_dataset_from_path(run_dir)
    return model, dataset


def load_test_accuracy_from_evaluation(run_dir, base_dir: str = "outputs"):
    """Load test accuracy from evaluation results."""
    eval_dir = Path(f"{base_dir}/{run_dir}/evaluation")

    if not eval_dir.exists():
        return None

    # Look for evaluation metrics file
    metrics_file = eval_dir / "evaluation_metrics.txt"
    if metrics_file.exists():
        try:
            with open(metrics_file, "r") as f:
                content = f.read()
                # Look for "Accuracy: X.XXXX" pattern
                for line in content.split("\n"):
                    if line.startswith("Accuracy:"):
                        accuracy_str = line.split(":")[1].strip()
                        return float(accuracy_str)
        except Exception as e:
            print(f"Error reading {metrics_file}: {e}")

    # Look for JSON evaluation results
    json_file = eval_dir / "evaluation_results.json"
    if json_file.exists():
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                if "accuracy" in data:
                    return float(data["accuracy"])
        except Exception as e:
            print(f"Error reading {json_file}: {e}")

    return None


def load_training_history(metadata_file):
    """Load training history from metadata file."""
    try:
        data = load_json_data(metadata_file)

        if "training_history" in data:
            history = data["training_history"]
            if "train_acc" in history and "val_acc" in history:
                train_acc = history["train_acc"]
                val_acc = history["val_acc"]

                if train_acc and val_acc:
                    # Get the last epoch values
                    last_train_acc = train_acc[-1] if train_acc else None
                    last_val_acc = val_acc[-1] if val_acc else None
                    return last_train_acc, last_val_acc, len(train_acc)

        return None, None, None
    except Exception as e:
        print(f"Error loading {metadata_file}: {e}")
        return None, None, None


def find_best_models(base_dir: str = "outputs"):
    """Find the best performing model for each model type and dataset based on test accuracy."""
    outputs_dir = Path(base_dir)
    model_results = {}

    # Find all model directories
    for run_dir in outputs_dir.iterdir():
        if run_dir.is_dir() and run_dir.name not in ["analysis", "aiccc_2025"]:
            model, dataset = extract_model_info(run_dir.name)
            if not model or not dataset:
                continue

            # Get test accuracy from evaluation results
            test_acc = load_test_accuracy_from_evaluation(run_dir.name, base_dir)

            if test_acc is not None:
                key = (model, dataset)
                if key not in model_results or test_acc > model_results[key]["test_acc"]:
                    model_results[key] = {"run_dir": run_dir.name, "test_acc": test_acc}

    return model_results


def create_overfitting_analysis(base_dir: str = "outputs"):
    """Create comprehensive overfitting analysis."""
    print("Finding best performing models...")
    best_models = find_best_models(base_dir)

    if not best_models:
        print("No model results found!")
        return None

    print(f"Found {len(best_models)} best models")

    # Create analysis data
    analysis_data = []

    for (model, dataset), info in best_models.items():
        run_dir = info["run_dir"]
        test_acc = info["test_acc"]

        print(f"Processing {model}-{dataset} (test_acc: {test_acc:.4f})")

        # Load training history
        metadata_file = Path(f"{base_dir}/{run_dir}/model_metadata.json")
        train_acc, val_acc, epochs = load_training_history(metadata_file)

        if train_acc is not None and val_acc is not None:
            overfitting_gap = train_acc - val_acc

            analysis_data.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "run_dir": run_dir,
                    "test_acc": test_acc,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "overfitting": overfitting_gap,
                    "epochs": epochs,
                }
            )
        else:
            print(f"  Warning: Could not load training history for {model}-{dataset}")

    if not analysis_data:
        print("No complete analysis data found!")
        return None

    # Create DataFrame
    df = pd.DataFrame(analysis_data)

    # Add percentage columns for display
    df["Test Acc (%)"] = (df["test_acc"] * 100).round(1)
    df["Training Acc (%)"] = (df["train_acc"] * 100).round(1)
    df["Validation Acc (%)"] = (df["val_acc"] * 100).round(1)
    df["Overfitting Gap (%)"] = (df["overfitting"] * 100).round(1)

    return df


def create_final_formatted_table(analysis_df):
    """Create final formatted table in specified order."""
    if analysis_df is None:
        return None

    # Define the exact order
    model_dataset_order = [
        ("FC", "FMA"),
        ("FC", "GTZAN"),
        ("CNN", "FMA"),
        ("CNN", "GTZAN"),
        ("LSTM", "FMA"),
        ("LSTM", "GTZAN"),
        ("XLSTM", "FMA"),
        ("XLSTM", "GTZAN"),
        ("GRU", "FMA"),
        ("GRU", "GTZAN"),
        ("TRANSFORMER", "FMA"),
        ("TRANSFORMER", "GTZAN"),
        ("VGG", "FMA"),
        ("VGG", "GTZAN"),
    ]

    # Create empty list to store rows
    final_rows = []

    # Process each model-dataset combination in the specified order
    for model, dataset in model_dataset_order:
        # Find the row for this model-dataset combination
        model_row = analysis_df[
            (analysis_df["model"] == model) & (analysis_df["dataset"] == dataset)
        ]

        if not model_row.empty:
            # Get the data
            row_data = {
                "Model": model,
                "Dataset": dataset,
                "Test Acc (%)": model_row["Test Acc (%)"].iloc[0],
                "Training Acc (%)": model_row["Training Acc (%)"].iloc[0],
                "Validation Acc (%)": model_row["Validation Acc (%)"].iloc[0],
                "Overfitting Gap (%)": model_row["Overfitting Gap (%)"].iloc[0],
            }
        else:
            # Create empty row for missing data
            row_data = {
                "Model": model,
                "Dataset": dataset,
                "Test Acc (%)": "N/A",
                "Training Acc (%)": "N/A",
                "Validation Acc (%)": "N/A",
                "Overfitting Gap (%)": "N/A",
            }

        final_rows.append(row_data)

    # Create final DataFrame
    final_df = pd.DataFrame(final_rows)
    return final_df


def print_final_table(df):
    """Print the final table in a formatted way."""
    if df is None:
        print("No data to display")
        return

    print(f"\n{'='*120}")
    print("OVERFITTING ANALYSIS - FINAL RESULTS")
    print(f"{'='*120}")

    # Header
    header = f"{'Model':<12} {'Dataset':<6} {'Test Acc (%)':<12} {'Training Acc (%)':<15} {'Validation Acc (%)':<15} {'Overfitting Gap (%)':<12}"
    print(header)
    print("-" * 120)

    # Print data rows
    for _, row in df.iterrows():
        test_acc = f"{row['Test Acc (%)']}" if row["Test Acc (%)"] != "N/A" else "N/A"
        train_acc = f"{row['Training Acc (%)']}" if row["Training Acc (%)"] != "N/A" else "N/A"
        val_acc = f"{row['Validation Acc (%)']}" if row["Validation Acc (%)"] != "N/A" else "N/A"
        overfitting = (
            f"{row['Overfitting Gap (%)']}" if row["Overfitting Gap (%)"] != "N/A" else "N/A"
        )

        line = f"{row['Model']:<12} {row['Dataset']:<6} {test_acc:<12} {train_acc:<15} {val_acc:<15} {overfitting:<12}"
        print(line)


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Overfitting Analysis Tool")
    parser.add_argument(
        "--input-dir",
        default="./outputs",
        help="Directory containing training results to analyze (default: ./outputs)",
    )
    parser.add_argument(
        "--output-dir",
        default="./outputs/analysis",
        help="Directory where analysis results will be saved (default: ./outputs/analysis)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("COMPREHENSIVE OVERFITTING ANALYSIS")
    print("=" * 60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")

    # Create output directory
    ensure_output_directory(args.output_dir)

    # Step 1: Create overfitting analysis
    print("\nStep 1: Generating overfitting analysis...")
    analysis_df = create_overfitting_analysis(args.input_dir)

    if analysis_df is None:
        print("Failed to generate overfitting analysis!")
        return 1

    # Save raw analysis
    raw_output_file = f"{args.output_dir}/overfitting_analysis.csv"
    save_dataframe(analysis_df, raw_output_file)
    print(f"Raw analysis saved to: {raw_output_file}")

    # Step 2: Create final formatted table
    print("\nStep 2: Creating final formatted table...")
    final_df = create_final_formatted_table(analysis_df)

    if final_df is None:
        print("Failed to create final table!")
        return 1

    # Print the table
    print_final_table(final_df)

    # Save final table
    final_output_file = f"{args.output_dir}/overfitting_analysis_final.csv"
    save_dataframe(final_df, final_output_file)
    print(f"\nFinal table saved to: {final_output_file}")

    # Print summary
    print(f"\n{'='*120}")
    print("SUMMARY")
    print(f"{'='*120}")
    print(f"Total rows: {len(final_df)}")
    print(f"Models included: {final_df['Model'].nunique()}")
    print(f"Datasets included: {final_df['Dataset'].nunique()}")

    # Count available data
    available_data = final_df[final_df["Overfitting Gap (%)"] != "N/A"]
    print(f"Rows with data: {len(available_data)}")
    print(f"Rows missing data: {len(final_df) - len(available_data)}")

    # Show test accuracy range
    test_accs = final_df[final_df["Test Acc (%)"] != "N/A"]["Test Acc (%)"]
    if len(test_accs) > 0:
        print(f"Test accuracy range: {test_accs.min():.1f}% to {test_accs.max():.1f}%")

    # Show overfitting statistics
    overfitting_gaps = final_df[final_df["Overfitting Gap (%)"] != "N/A"]["Overfitting Gap (%)"]
    if len(overfitting_gaps) > 0:
        print(
            f"Overfitting gap range: {overfitting_gaps.min():.1f}% to {overfitting_gaps.max():.1f}%"
        )
        print(f"Average overfitting gap: {overfitting_gaps.mean():.1f}%")

    print("\nAnalysis completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
