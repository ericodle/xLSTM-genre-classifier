"""
Model evaluation module for GenreDiscern.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from scipy.stats import ks_2samp
import logging
import os

from core.utils import setup_logging, get_device
from models import BaseModel
from core.constants import GTZAN_GENRES


class ModelEvaluator:
    """Handles model evaluation and performance analysis."""

    def __init__(
        self,
        model: BaseModel,
        device: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.model = model
        self.device = get_device(device or "auto")
        self.logger = logger or setup_logging()

        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()

        self.logger.info(f"ModelEvaluator initialized on device: {self.device}")

    def evaluate_model(
        self, test_loader: DataLoader, class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the model on test data.

        Args:
            test_loader: DataLoader for test data
            class_names: List of class names for reporting

        Returns:
            Dictionary containing evaluation results
        """
        self.logger.info("Starting model evaluation...")

        # Get predictions and ground truth
        y_true, y_pred, y_probs = self._get_predictions(test_loader)

        # Calculate metrics
        results = {}

        # Basic accuracy
        accuracy = (y_true == y_pred).mean()
        results["accuracy"] = accuracy

        # Classification report
        if class_names:
            results["classification_report"] = classification_report(
                y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
            )
        else:
            results["classification_report"] = classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            )

        # Confusion matrix
        results["confusion_matrix"] = confusion_matrix(y_true, y_pred)

        # ROC AUC scores
        if y_probs.shape[1] == 2:  # Binary classification
            results["roc_auc"] = roc_auc_score(y_true, y_probs[:, 1])
        else:  # Multi-class classification
            # Ensure probabilities sum to 1.0 for each sample
            y_probs_normalized = y_probs / y_probs.sum(axis=1, keepdims=True)
            try:
                results["roc_auc"] = roc_auc_score(
                    y_true, y_probs_normalized, multi_class="ovr", average="macro"
                )
            except ValueError as e:
                self.logger.warning(f"ROC AUC calculation failed: {e}")
                results["roc_auc"] = 0.0

        # KS test statistics
        results["ks_stats"] = self._calculate_ks_test(y_true, y_probs)

        # Precision-Recall curves
        results["precision_recall"] = self._calculate_precision_recall(y_true, y_probs_normalized if y_probs.shape[1] > 2 else y_probs)

        self.logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}")

        return results

    def _get_predictions(
        self, test_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get model predictions and ground truth."""
        y_true = []
        y_pred = []
        y_probs = []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                output = self.model(data)

                # Get probabilities and predictions
                if hasattr(output, "log_softmax"):
                    # Handle log_softmax output
                    probs = torch.exp(output)
                else:
                    probs = torch.softmax(output, dim=1)

                pred = torch.argmax(probs, dim=1)

                # Store results
                y_true.append(target.cpu().numpy())
                y_pred.append(pred.cpu().numpy())
                y_probs.append(probs.cpu().numpy())

        # Concatenate all batches
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        y_probs = np.concatenate(y_probs)

        return y_true, y_pred, y_probs

    def _calculate_ks_test(
        self, y_true: np.ndarray, y_probs: np.ndarray
    ) -> List[float]:
        """
        Calculate Kolmogorov-Smirnov test statistics for each class.

        Args:
            y_true: Ground truth labels
            y_probs: Predicted probabilities

        Returns:
            List of KS statistics for each class
        """
        ks_test_stats: list[float] = []
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

        return ks_test_stats

    def _calculate_precision_recall(
        self, y_true: np.ndarray, y_probs: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate precision-recall curves and average precision."""
        results = {}

        if y_probs.shape[1] == 2:  # Binary classification
            # Binary case
            precision, recall, _ = precision_recall_curve(y_true, y_probs[:, 1])
            avg_precision = average_precision_score(y_true, y_probs[:, 1])

            results["binary"] = {
                "precision": precision,
                "recall": recall,
                "average_precision": avg_precision,
            }
        else:
            # Multi-class case
            results["multiclass"] = {}
            for i in range(y_probs.shape[1]):
                precision, recall, _ = precision_recall_curve(
                    y_true == i, y_probs[:, i]
                )
                avg_precision = average_precision_score(y_true == i, y_probs[:, i])

                results["multiclass"][f"class_{i}"] = {
                    "precision": precision,
                    "recall": recall,
                    "average_precision": avg_precision,
                }

        return results

    def generate_evaluation_plots(
        self,
        results: Dict[str, Any],
        output_dir: str,
        class_names: Optional[List[str]] = None,
    ) -> None:
        """
        Generate evaluation visualization plots.

        Args:
            results: Evaluation results from evaluate_model
            output_dir: Directory to save plots
            class_names: List of class names for labels
        """
        os.makedirs(output_dir, exist_ok=True)

        # Confusion matrix plot
        self._plot_confusion_matrix(
            results["confusion_matrix"],
            class_names,
            os.path.join(output_dir, "confusion_matrix.png"),
        )

        # ROC curves
        if "roc_auc" in results:
            self._plot_roc_curves(
                results, class_names, os.path.join(output_dir, "roc_curves.png")
            )

        # KS curves
        if "ks_stats" in results:
            self._plot_ks_curves(
                results, class_names, os.path.join(output_dir, "ks_curves.png")
            )

        # Precision-Recall curves
        self._plot_precision_recall_curves(
            results["precision_recall"],
            class_names,
            os.path.join(output_dir, "precision_recall_curves.png"),
        )

        self.logger.info(f"Evaluation plots saved to: {output_dir}")

    def _plot_confusion_matrix(
        self, cm: np.ndarray, class_names: Optional[List[str]], output_path: str
    ):
        """Plot confusion matrix."""
        plt.figure(figsize=(10, 8))

        if class_names:
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
            )
        else:
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_roc_curves(
        self,
        results: Dict[str, Any],
        class_names: Optional[List[str]],
        output_path: str,
    ):
        """Plot ROC curves."""
        plt.figure(figsize=(10, 8))

        # This is a placeholder - actual ROC curve plotting would need
        # the raw model outputs before softmax
        plt.text(
            0.5,
            0.5,
            f"ROC AUC: {results['roc_auc']:.4f}",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
            fontsize=16,
        )
        plt.title("ROC Curves")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_ks_curves(
        self,
        results: Dict[str, Any],
        class_names: Optional[List[str]],
        output_path: str,
    ):
        """Plot KS curves."""
        plt.figure(figsize=(10, 8))

        # Get predictions for plotting
        y_true, y_pred, y_probs = self._get_predictions_from_results(results)

        if y_probs is not None:
            for class_idx in range(y_probs.shape[1]):
                # Get probabilities for samples that actually belong to this class
                class_true = y_probs[y_true == class_idx, class_idx]

                if len(class_true) > 0:
                    class_name = (
                        class_names[class_idx] if class_names else f"Class {class_idx}"
                    )
                    ks_stat = results["ks_stats"][class_idx]

                    # Sort probabilities for plotting
                    sorted_probs = np.sort(class_true)
                    plt.plot(sorted_probs, label=f"{class_name} (KS = {ks_stat:.3f})")

        plt.xlabel("Sorted Class Probabilities")
        plt.ylabel("CDF")
        plt.title("KS Curves")
        if len(plt.gca().get_lines()) > 0:  # Only add legend if there are lines to show
            plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _get_predictions_from_results(
        self, results: Dict[str, Any]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Helper method to get predictions from results if available."""
        # This would need to be implemented based on how results are stored
        # For now, return None values
        return None, None, None

    def _plot_precision_recall_curves(
        self,
        precision_recall: Dict[str, Any],
        class_names: Optional[List[str]],
        output_path: str,
    ):
        """Plot precision-recall curves."""
        plt.figure(figsize=(10, 8))

        if "binary" in precision_recall:
            # Binary case
            pr_data = precision_recall["binary"]
            plt.plot(
                pr_data["recall"],
                pr_data["precision"],
                label=f'AP = {pr_data["average_precision"]:.3f}',
            )
        else:
            # Multi-class case
            for class_idx, pr_data in precision_recall["multiclass"].items():
                # Extract class index from "class_X" format
                if "_" in class_idx:
                    class_num = int(class_idx.split("_")[1])
                    class_name = (
                        class_names[str(class_num)]
                        if class_names and str(class_num) in class_names
                        else f"Class {class_num}"
                    )
                else:
                    class_name = class_idx
                plt.plot(
                    pr_data["recall"],
                    pr_data["precision"],
                    label=f'{class_name} (AP = {pr_data["average_precision"]:.3f})',
                )

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curves")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def save_evaluation_results(
        self, results: Dict[str, Any], output_path: str
    ) -> None:
        """Save evaluation results to file."""
        import json

        def convert_to_serializable(obj):
            """Recursively convert numpy arrays and other non-serializable objects to JSON-serializable types."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        # Convert all results to JSON-serializable format
        serializable_results = convert_to_serializable(results)

        with open(output_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        self.logger.info(f"Evaluation results saved to: {output_path}")

    def get_model_summary(self) -> Dict[str, Any]:
        """Get a summary of the model."""
        return self.model.get_model_info()
