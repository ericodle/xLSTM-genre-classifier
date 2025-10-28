import logging
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve


class PlottingUtilities:
    """Utilities for creating evaluation plots and visualizations."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize plotting utilities."""
        self.logger = logger or logging.getLogger(__name__)

    @staticmethod
    def plot_confusion_matrix(
        cm: np.ndarray,
        class_names: Optional[List[str]],
        output_path: str,
        title: str = "Confusion Matrix",
    ) -> None:
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

        plt.title(title)
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_roc_curves(
        results: Dict,
        class_names: Optional[List[str]],
        output_path: str,
        title: str = "ROC Curves",
    ) -> None:
        """Plot ROC curves."""
        plt.figure(figsize=(10, 8))

        y_true = results.get("y_true")
        y_probs = results.get("y_probs")

        if y_probs is not None and len(y_probs) > 0:
            if y_probs.shape[1] == 2:  # Binary classification
                fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
                roc_auc = results.get("roc_auc", 0.0)
                plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
            else:  # Multi-class classification
                for class_idx in range(y_probs.shape[1]):
                    y_true_binary = (y_true == class_idx).astype(int)
                    if len(np.unique(y_true_binary)) > 1:
                        fpr, tpr, _ = roc_curve(y_true_binary, y_probs[:, class_idx])
                        class_name = class_names[class_idx] if class_names else f"Class {class_idx}"
                        auc = roc_auc_score(y_true_binary, y_probs[:, class_idx])
                        plt.plot(fpr, tpr, label=f"{class_name} (AUC = {auc:.3f})")

            plt.plot([0, 1], [0, 1], "k--", label="Random classifier")
            plt.legend(loc="lower right")
        else:
            plt.text(
                0.5,
                0.5,
                f"ROC AUC: {results.get('roc_auc', 0.0):.4f}\n(No detailed curves available)",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
                fontsize=16,
            )

        plt.title(title)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_ks_curves(
        results: Dict,
        class_names: Optional[List[str]],
        output_path: str,
        title: str = "Kolmogorov-Smirnov Test Analysis",
    ) -> None:
        """Plot KS test visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16, fontweight="bold")

        y_true = results.get("y_true")
        y_probs = results.get("y_probs")
        ks_stats = results.get("ks_stats", [])

        if y_probs is not None and len(y_probs) > 0 and len(ks_stats) > 0:
            class_labels = [
                class_names[i] if class_names and i < len(class_names) else f"Class {i}"
                for i in range(len(ks_stats))
            ]

            # 1. KS Statistics Bar Chart
            ax1 = axes[0, 0]
            bars = ax1.bar(
                range(len(ks_stats)), ks_stats, color="skyblue", edgecolor="navy", alpha=0.7
            )
            ax1.set_xlabel("Class")
            ax1.set_ylabel("KS Statistic")
            ax1.set_title("KS Statistics by Class")
            ax1.set_xticks(range(len(ks_stats)))
            ax1.set_xticklabels(class_labels, rotation=45, ha="right")
            ax1.grid(True, alpha=0.3)

            for i, (bar, stat) in enumerate(zip(bars, ks_stats)):
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{stat:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            # 2. KS Test Visualization (CDFs)
            ax2 = axes[0, 1]
            for class_idx in range(y_probs.shape[1]):
                class_true = y_probs[y_true == class_idx, class_idx]
                class_all = y_probs[:, class_idx]

                if len(class_true) > 0:
                    class_name = class_labels[class_idx]
                    sorted_true = np.sort(class_true)
                    sorted_all = np.sort(class_all)
                    n_true = len(sorted_true)
                    n_all = len(sorted_all)
                    cdf_true = np.arange(1, n_true + 1) / n_true
                    cdf_all = np.arange(1, n_all + 1) / n_all

                    ax2.plot(sorted_true, cdf_true, label=f"{class_name} (True)", linewidth=2)
                    ax2.plot(sorted_all, cdf_all, "--", label=f"{class_name} (All)", linewidth=2)

            ax2.set_xlabel("Class Probability")
            ax2.set_ylabel("Cumulative Distribution Function")
            ax2.set_title("CDF Comparison (True vs All Samples)")
            ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax2.grid(True, alpha=0.3)

            # 3. Distribution Comparison (Histograms)
            ax3 = axes[1, 0]
            for class_idx in range(min(3, y_probs.shape[1])):
                class_true = y_probs[y_true == class_idx, class_idx]
                class_all = y_probs[:, class_idx]

                if len(class_true) > 0:
                    class_name = class_labels[class_idx]
                    ax3.hist(
                        class_true, bins=30, alpha=0.6, label=f"{class_name} (True)", density=True
                    )
                    ax3.hist(
                        class_all, bins=30, alpha=0.3, label=f"{class_name} (All)", density=True
                    )

            ax3.set_xlabel("Class Probability")
            ax3.set_ylabel("Density")
            ax3.set_title("Probability Distribution Comparison")
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # 4. KS Statistics Summary
            ax4 = axes[1, 1]
            ax4.axis("off")

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

            ax4.text(
                0.05,
                0.95,
                summary_text,
                transform=ax4.transAxes,
                fontsize=10,
                verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def create_metrics_table(
        results: Dict,
        class_names: Optional[List[str]],
        output_path: str,
        title: str = "Classification Metrics",
    ) -> None:
        """Create a comprehensive metrics table."""
        plt.figure(figsize=(12, 8))

        classification_report = results.get("classification_report", {})

        if not classification_report:
            plt.text(
                0.5,
                0.5,
                "No classification report available",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )
            plt.title(title)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            return

        metrics_data = []
        headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]

        for class_name, metrics in classification_report.items():
            if isinstance(metrics, dict) and "precision" in metrics:
                class_display_name = class_name
                if class_names and str(class_name).isdigit():
                    class_idx = int(class_name)
                    if class_idx < len(class_names):
                        class_display_name = class_names[class_idx]

                metrics_data.append(
                    [
                        class_display_name,
                        f"{metrics['precision']:.3f}",
                        f"{metrics['recall']:.3f}",
                        f"{metrics['f1-score']:.3f}",
                        f"{int(metrics['support'])}",
                    ]
                )

        if "macro avg" in classification_report:
            metrics_data.append(
                [
                    "Macro Avg",
                    f"{classification_report['macro avg']['precision']:.3f}",
                    f"{classification_report['macro avg']['recall']:.3f}",
                    f"{classification_report['macro avg']['f1-score']:.3f}",
                    f"{int(classification_report['macro avg']['support'])}",
                ]
            )

        if "weighted avg" in classification_report:
            metrics_data.append(
                [
                    "Weighted Avg",
                    f"{classification_report['weighted avg']['precision']:.3f}",
                    f"{classification_report['weighted avg']['recall']:.3f}",
                    f"{classification_report['weighted avg']['f1-score']:.3f}",
                    f"{int(classification_report['weighted avg']['support'])}",
                ]
            )

        if metrics_data:
            table = plt.table(
                cellText=metrics_data,
                colLabels=headers,
                cellLoc="center",
                loc="center",
                bbox=[0, 0, 1, 1],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)

            for i in range(len(headers)):
                table[(0, i)].set_facecolor("#40466e")
                table[(0, i)].set_text_props(weight="bold", color="white")

            for i in range(1, len(metrics_data) + 1):
                for j in range(len(headers)):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor("#f1f1f2")
                    else:
                        table[(i, j)].set_facecolor("white")

        plt.axis("off")
        plt.title(title, fontsize=16, fontweight="bold", pad=20)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
