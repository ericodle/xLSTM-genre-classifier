import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.stats import ks_2samp
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader

from core.utils import get_device, setup_logging
from eval.model_loader import UnifiedModelLoader
from eval.plotting_utils import PlottingUtilities
from models import BaseModel


class ModelEvaluator:
    """Handles model evaluation and performance analysis."""

    def __init__(
        self,
        model: Union[BaseModel, str],
        device: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger or setup_logging()
        self.device = get_device(device or "auto")

        # Handle both model objects and model paths
        if isinstance(model, str):
            # Load model from path
            loader = UnifiedModelLoader(logger=self.logger)
            self.model = loader.load_model(model)
        else:
            # Use provided model object
            self.model = model
            # Move to device if it's a PyTorch model
            if hasattr(self.model, "to"):
                self.model = self.model.to(self.device)

        # Set to evaluation mode
        self.model.eval()

        # Initialize plotting utilities
        self.plotting = PlottingUtilities(logger=self.logger)

        self.logger.info(f"ModelEvaluator initialized on device: {self.device}")
        self.logger.info(f"Model type: {getattr(self.model, 'model_type', 'Unknown')}")

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

    def _calculate_ks_test(self, y_true: np.ndarray, y_probs: np.ndarray) -> List[float]:
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
        self.plotting.plot_confusion_matrix(
            results["confusion_matrix"],
            class_names,
            os.path.join(output_dir, "confusion_matrix.png"),
        )

        # ROC curves
        if "roc_auc" in results:
            self.plotting.plot_roc_curves(
                results, class_names, os.path.join(output_dir, "roc_curves.png")
            )

        # KS curves
        if "ks_stats" in results:
            self.plotting.plot_ks_curves(
                results, class_names, os.path.join(output_dir, "ks_curves.png")
            )

        # Metrics table
        self.plotting.create_metrics_table(
            results,
            class_names,
            os.path.join(output_dir, "metrics_table.png"),
        )

        self.logger.info(f"Evaluation plots saved to: {output_dir}")

    def save_evaluation_results(self, results: Dict[str, Any], output_path: str) -> None:
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
