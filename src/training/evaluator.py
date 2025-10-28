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

        # Note: Precision-Recall curves removed in favor of metrics table

        # Store predictions for plotting
        results["y_true"] = y_true
        results["y_pred"] = y_pred
        results["y_probs"] = y_probs

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

        # Metrics table
        self._create_metrics_table(
            results,
            class_names,
            os.path.join(output_dir, "metrics_table.png"),
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

        # Get predictions for plotting
        y_true, y_pred, y_probs = self._get_predictions_from_results(results)
        
        if y_probs is not None and len(y_probs) > 0:
            if y_probs.shape[1] == 2:  # Binary classification
                # Binary ROC curve
                fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
                roc_auc = results.get('roc_auc', 0.0)
                plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
            else:  # Multi-class classification
                # Plot ROC curve for each class
                for class_idx in range(y_probs.shape[1]):
                    # Binarize the labels for this class
                    y_true_binary = (y_true == class_idx).astype(int)
                    if len(np.unique(y_true_binary)) > 1:  # Check if class exists in test set
                        fpr, tpr, _ = roc_curve(y_true_binary, y_probs[:, class_idx])
                        class_name = class_names[class_idx] if class_names else f"Class {class_idx}"
                        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc_score(y_true_binary, y_probs[:, class_idx]):.3f})')
            
            # Plot diagonal line for reference
            plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
            plt.legend(loc="lower right")
        else:
            # Fallback if no data available
            plt.text(
                0.5,
                0.5,
                f"ROC AUC: {results.get('roc_auc', 0.0):.4f}\n(No detailed curves available)",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
                fontsize=16,
            )

        plt.title("ROC Curves")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_ks_curves(
        self,
        results: Dict[str, Any],
        class_names: Optional[List[str]],
        output_path: str,
    ):
        """Plot KS test visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Kolmogorov-Smirnov Test Analysis", fontsize=16, fontweight='bold')

        # Get predictions for plotting
        y_true, y_pred, y_probs = self._get_predictions_from_results(results)

        if y_probs is not None and len(y_probs) > 0:
            # 1. KS Statistics Bar Chart
            ax1 = axes[0, 0]
            ks_stats = results["ks_stats"]
            class_labels = [class_names[i] if class_names and i < len(class_names) else f"Class {i}" 
                           for i in range(len(ks_stats))]
            
            bars = ax1.bar(range(len(ks_stats)), ks_stats, color='skyblue', edgecolor='navy', alpha=0.7)
            ax1.set_xlabel('Class')
            ax1.set_ylabel('KS Statistic')
            ax1.set_title('KS Statistics by Class')
            ax1.set_xticks(range(len(ks_stats)))
            ax1.set_xticklabels(class_labels, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, stat) in enumerate(zip(bars, ks_stats)):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{stat:.3f}', ha='center', va='bottom', fontsize=9)

            # 2. KS Test Visualization (CDFs with max difference)
            ax2 = axes[0, 1]
            for class_idx in range(y_probs.shape[1]):
                class_true = y_probs[y_true == class_idx, class_idx]
                class_all = y_probs[:, class_idx]
                
                if len(class_true) > 0:
                    class_name = class_labels[class_idx]
                    ks_stat = results["ks_stats"][class_idx]
                    
                    # Sort probabilities for CDF
                    sorted_true = np.sort(class_true)
                    sorted_all = np.sort(class_all)
                    
                    # Create CDFs
                    n_true = len(sorted_true)
                    n_all = len(sorted_all)
                    cdf_true = np.arange(1, n_true + 1) / n_true
                    cdf_all = np.arange(1, n_all + 1) / n_all
                    
                    # Plot CDFs
                    ax2.plot(sorted_true, cdf_true, label=f'{class_name} (True)', linewidth=2)
                    ax2.plot(sorted_all, cdf_all, '--', label=f'{class_name} (All)', linewidth=2)
            
            ax2.set_xlabel('Class Probability')
            ax2.set_ylabel('Cumulative Distribution Function')
            ax2.set_title('CDF Comparison (True vs All Samples)')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)

            # 3. Distribution Comparison (Histograms)
            ax3 = axes[1, 0]
            for class_idx in range(min(3, y_probs.shape[1])):  # Show first 3 classes
                class_true = y_probs[y_true == class_idx, class_idx]
                class_all = y_probs[:, class_idx]
                
                if len(class_true) > 0:
                    class_name = class_labels[class_idx]
                    ks_stat = results["ks_stats"][class_idx]
                    
                    # Plot histograms
                    ax3.hist(class_true, bins=30, alpha=0.6, label=f'{class_name} (True)', density=True)
                    ax3.hist(class_all, bins=30, alpha=0.3, label=f'{class_name} (All)', density=True)
            
            ax3.set_xlabel('Class Probability')
            ax3.set_ylabel('Density')
            ax3.set_title('Probability Distribution Comparison')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # 4. KS Statistics Summary
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            # Create summary text
            summary_text = "KS Test Summary:\n\n"
            summary_text += f"Number of classes: {len(ks_stats)}\n"
            summary_text += f"Mean KS statistic: {np.mean(ks_stats):.3f}\n"
            summary_text += f"Max KS statistic: {np.max(ks_stats):.3f}\n"
            summary_text += f"Min KS statistic: {np.min(ks_stats):.3f}\n\n"
            
            summary_text += "Interpretation:\n"
            summary_text += "• Higher KS = Better discrimination\n"
            summary_text += "• KS > 0.2 = Good discrimination\n"
            summary_text += "• KS > 0.4 = Excellent discrimination\n"
            summary_text += "• KS < 0.1 = Poor discrimination\n\n"
            
            summary_text += "Top performing classes:\n"
            sorted_indices = np.argsort(ks_stats)[::-1]
            for i, idx in enumerate(sorted_indices[:3]):
                if ks_stats[idx] > 0:
                    summary_text += f"{i+1}. {class_labels[idx]}: {ks_stats[idx]:.3f}\n"
            
            ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _get_predictions_from_results(
        self, results: Dict[str, Any]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Helper method to get predictions from results if available."""
        # Try to get predictions from results dictionary
        y_true = results.get('y_true')
        y_pred = results.get('y_pred') 
        y_probs = results.get('y_probs')
        
        return y_true, y_pred, y_probs

    def _create_metrics_table(
        self,
        results: Dict[str, Any],
        class_names: Optional[List[str]],
        output_path: str,
    ):
        """Create a comprehensive metrics table."""
        plt.figure(figsize=(12, 8))
        
        # Get classification report
        classification_report = results.get("classification_report", {})
        
        if not classification_report:
            plt.text(0.5, 0.5, "No classification report available", 
                    ha="center", va="center", transform=plt.gca().transAxes)
            plt.title("Metrics Table")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            return
        
        # Prepare data for table
        metrics_data = []
        headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
        
        # Add per-class metrics
        for class_name, metrics in classification_report.items():
            if isinstance(metrics, dict) and "precision" in metrics:
                # This is a class entry
                class_display_name = class_name
                if class_names and class_name.isdigit():
                    class_idx = int(class_name)
                    if class_idx < len(class_names):
                        class_display_name = class_names[class_idx]
                
                metrics_data.append([
                    class_display_name,
                    f"{metrics['precision']:.3f}",
                    f"{metrics['recall']:.3f}",
                    f"{metrics['f1-score']:.3f}",
                    f"{int(metrics['support'])}"
                ])
        
        # Add macro and weighted averages
        if "macro avg" in classification_report:
            metrics_data.append([
                "Macro Avg",
                f"{classification_report['macro avg']['precision']:.3f}",
                f"{classification_report['macro avg']['recall']:.3f}",
                f"{classification_report['macro avg']['f1-score']:.3f}",
                f"{int(classification_report['macro avg']['support'])}"
            ])
        
        if "weighted avg" in classification_report:
            metrics_data.append([
                "Weighted Avg",
                f"{classification_report['weighted avg']['precision']:.3f}",
                f"{classification_report['weighted avg']['recall']:.3f}",
                f"{classification_report['weighted avg']['f1-score']:.3f}",
                f"{int(classification_report['weighted avg']['support'])}"
            ])
        
        # Create table
        if metrics_data:
            table = plt.table(
                cellText=metrics_data,
                colLabels=headers,
                cellLoc='center',
                loc='center',
                bbox=[0, 0, 1, 1]
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # Style the table
            for i in range(len(headers)):
                table[(0, i)].set_facecolor('#40466e')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Alternate row colors
            for i in range(1, len(metrics_data) + 1):
                for j in range(len(headers)):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#f1f1f2')
                    else:
                        table[(i, j)].set_facecolor('white')
        
        plt.axis('off')
        plt.title("Classification Metrics", fontsize=16, fontweight='bold', pad=20)
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
