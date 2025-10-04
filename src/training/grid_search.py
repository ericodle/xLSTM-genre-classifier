"""
Grid search hyperparameter optimization for GenreDiscern.
"""

import os
import json
import time
import itertools
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import logging

from core.utils import setup_logging, ensure_directory
from core.config import Config
from training.trainer import ModelTrainer
from training.trainer_unified import UnifiedTrainer


class GridSearchTrainer:
    """Grid search hyperparameter optimization."""

    def __init__(
        self, config: Optional[Config] = None, logger: Optional[logging.Logger] = None
    ):
        self.config = config or Config()
        self.logger = logger or setup_logging()
        self.results: list[dict[str, Any]] = []
        self.status_file = None
        self.completed_combinations = set()

    def run_grid_search(
        self,
        data_path: str,
        model_type: str,
        base_output_dir: str,
        param_grid: Dict[str, List[Any]],
        resume: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Run grid search over hyperparameters.

        Args:
            data_path: Path to the data file
            model_type: Type of model to train
            base_output_dir: Base directory for output
            param_grid: Dictionary of parameter names to lists of values
            resume: Whether to resume from existing results
            **kwargs: Additional arguments for training

        Returns:
            DataFrame with grid search results
        """
        self.logger.info(f"Starting grid search for {model_type} model")
        self.logger.info(f"Parameter grid: {param_grid}")

        # Generate all parameter combinations
        param_combinations = list(itertools.product(*param_grid.values()))
        param_names = list(param_grid.keys())

        total_combinations = len(param_combinations)
        self.logger.info(f"Total parameter combinations: {total_combinations}")

        # Create results directory
        ensure_directory(base_output_dir)
        
        # Setup status tracking
        self.status_file = os.path.join(base_output_dir, "grid_search_status.json")
        
        # Load existing results if resuming
        if resume:
            self._load_existing_results(base_output_dir)
            completed_count = len(self.completed_combinations)
            if completed_count > 0:
                self.logger.info(f"Found {completed_count} completed combinations, resuming...")
            else:
                self.logger.info("No existing results found, starting fresh")

        # Run training for each combination
        for i, param_values in enumerate(param_combinations):
            # Create parameter dictionary
            params = dict(zip(param_names, param_values))
            
            # Create parameter hash for tracking
            param_hash = self._create_param_hash(params)
            
            # Determine expected output directory for this combination (without creating it)
            expected_output_dir = os.path.join(base_output_dir, self._build_dir_name_from_params(params))
            
            # Skip if already completed AND the expected run directory actually exists
            if param_hash in self.completed_combinations and os.path.exists(expected_output_dir):
                self.logger.info(f"Skipping combination {i+1}/{total_combinations} (already completed)")
                continue
            elif param_hash in self.completed_combinations and not os.path.exists(expected_output_dir):
                # The status claims completed but the run directory is missing; treat as not completed
                self.logger.info(
                    f"Re-running combination {i+1}/{total_combinations} because expected directory is missing: {expected_output_dir}"
                )
                
            self.logger.info(f"Training combination {i+1}/{total_combinations}")

            # Create output directory for this combination
            output_dir = self._create_output_dir(base_output_dir, params)

            try:
                # Train model with these parameters
                result = self._train_single_combination(
                    data_path, model_type, output_dir, params, **kwargs
                )

                # Store results
                result.update(params)
                result["combination_id"] = i
                result["output_dir"] = output_dir
                result["param_hash"] = param_hash
                self.results.append(result)
                
                # Mark as completed
                self.completed_combinations.add(param_hash)
                
                # Save checkpoint after each successful combination
                self._save_checkpoint()

                self.logger.info(f"Combination {i+1} completed successfully")

            except Exception as e:
                self.logger.error(f"Combination {i+1} failed: {e}")
                # Store failure result
                failure_result = {
                    "combination_id": i,
                    "output_dir": output_dir,
                    "status": "failed",
                    "error": str(e),
                    "param_hash": param_hash,
                }
                failure_result.update(params)
                self.results.append(failure_result)
                
                # Mark as completed (even if failed)
                self.completed_combinations.add(param_hash)
                
                # Save checkpoint after each combination (success or failure)
                self._save_checkpoint()

        # Save results
        self._save_results(base_output_dir)

        # Generate and save comprehensive summary
        summary = self._generate_summary()
        summary_file = os.path.join(base_output_dir, "grid_search_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        # Generate detailed summary report
        self._generate_summary_report(base_output_dir, summary)

        # Generate visualization plots
        self._generate_visualization_plots(base_output_dir, summary)

        self.logger.info("Grid search completed")
        self.logger.info(f"Results saved to: {base_output_dir}")

        return pd.DataFrame(self.results)

    def _train_single_combination(
        self,
        data_path: str,
        model_type: str,
        output_dir: str,
        params: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        """Train a single parameter combination using unified trainer."""
        # Create unified trainer
        unified_trainer = UnifiedTrainer(self.config, self.logger)
        
        # Train with parameter overrides
        results = unified_trainer.train_with_parameters(
            data_path=data_path,
            model_type=model_type,
            output_dir=output_dir,
            parameters=params,
            **kwargs
        )
        
        # Extract metrics for grid search
        training_history = results["training_history"]
        test_loss = results["final_test_loss"]
        test_acc = results["final_test_accuracy"]

        # Get final metrics
        final_train_loss = (
            training_history["train_loss"][-1]
            if training_history["train_loss"]
            else float("inf")
        )
        final_val_loss = (
            training_history["val_loss"][-1]
            if training_history["val_loss"]
            else float("inf")
        )
        final_train_acc = (
            training_history["train_acc"][-1]
            if training_history["train_acc"]
            else 0.0
        )
        final_val_acc = (
            training_history["val_acc"][-1]
            if training_history["val_acc"]
            else 0.0
        )

        # Get best validation metrics
        best_val_loss = (
            min(training_history["val_loss"])
            if training_history["val_loss"]
            else float("inf")
        )
        best_val_acc = (
            max(training_history["val_acc"])
            if training_history["val_acc"]
            else 0.0
        )

        return {
            "status": "completed",
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "final_train_acc": final_train_acc,
            "final_val_acc": final_val_acc,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,  # Primary ranking metric
            "num_epochs": len(training_history["train_loss"]),
            "training_time": time.time(),  # Placeholder for actual training time
        }

    def _create_output_dir(self, base_dir: str, params: Dict[str, Any]) -> str:
        """Create output directory name based on parameters."""
        # Create a readable directory name
        dir_parts = []
        for param_name, param_value in params.items():
            # Convert parameter value to string, handling special cases
            if isinstance(param_value, float):
                param_str = f"{param_name}{param_value:.3f}".replace(".", "p")
            else:
                param_str = f"{param_name}{param_value}"
            dir_parts.append(param_str)

        dir_name = "_".join(dir_parts)
        output_dir = os.path.join(base_dir, dir_name)
        ensure_directory(output_dir)

        return output_dir

    def _save_results(self, output_dir: str) -> None:
        """Save grid search results to files."""
        # Save as JSON
        results_file = os.path.join(output_dir, "grid_search_results.json")
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)

        # Save as CSV
        df = pd.DataFrame(self.results)
        csv_file = os.path.join(output_dir, "grid_search_results.csv")
        df.to_csv(csv_file, index=False)

        self.logger.info(f"Results saved to {results_file} and {csv_file}")

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary of grid search results."""
        if not self.results:
            return {}

        # Filter successful runs
        successful_runs = [r for r in self.results if r.get("status") == "completed"]
        failed_runs = [r for r in self.results if r.get("status") == "failed"]

        if not successful_runs:
            return {"status": "all_failed", "failed_count": len(failed_runs)}

        # Find best parameters using test accuracy as primary metric
        best_test_acc_idx = np.argmax(
            [r.get("test_acc", 0) for r in successful_runs]
        )
        best_test_loss_idx = np.argmin(
            [r.get("test_loss", float("inf")) for r in successful_runs]
        )

        best_acc_run = successful_runs[best_test_acc_idx]
        best_loss_run = successful_runs[best_test_loss_idx]

        # Calculate statistics
        test_accuracies = [r.get("test_acc", 0) for r in successful_runs]
        test_losses = [r.get("test_loss", float("inf")) for r in successful_runs]
        val_accuracies = [r.get("best_val_acc", 0) for r in successful_runs]
        val_losses = [r.get("best_val_loss", float("inf")) for r in successful_runs]

        summary = {
            "total_combinations": len(self.results),
            "successful_runs": len(successful_runs),
            "failed_runs": len(failed_runs),
            "success_rate": len(successful_runs) / len(self.results),
            "best_accuracy": {
                "value": best_acc_run.get("test_acc", 0),
                "parameters": {
                    k: v
                    for k, v in best_acc_run.items()
                    if k not in ["status", "combination_id", "output_dir", "error"]
                },
            },
            "best_loss": {
                "value": best_loss_run.get("test_loss", float("inf")),
                "parameters": {
                    k: v
                    for k, v in best_loss_run.items()
                    if k not in ["status", "combination_id", "output_dir", "error"]
                },
            },
            "statistics": {
                "mean_test_acc": np.mean(test_accuracies),
                "std_test_acc": np.std(test_accuracies),
                "mean_test_loss": np.mean(test_losses),
                "std_test_loss": np.std(test_losses),
                "mean_val_acc": np.mean(val_accuracies),
                "std_val_acc": np.std(val_accuracies),
                "mean_val_loss": np.mean(val_losses),
                "std_val_loss": np.std(val_losses),
            },
        }

        return summary

    def get_best_parameters(self, metric: str = "accuracy") -> Optional[Dict[str, Any]]:
        """
        Get the best parameters based on a metric.

        Args:
            metric: Metric to optimize ('accuracy' or 'loss')

        Returns:
            Dictionary of best parameters
        """
        if not self.results:
            return None

        successful_runs = [r for r in self.results if r.get("status") == "completed"]
        if not successful_runs:
            return None

        if metric == "accuracy":
            best_run = max(successful_runs, key=lambda x: x.get("best_val_acc", 0))
        elif metric == "loss":
            best_run = min(
                successful_runs, key=lambda x: x.get("best_val_loss", float("inf"))
            )
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Extract only parameter values
        params = {}
        for key, value in best_run.items():
            if key not in [
                "status",
                "combination_id",
                "output_dir",
                "error",
                "final_train_loss",
                "final_val_loss",
                "final_train_acc",
                "final_val_acc",
                "best_val_loss",
                "best_val_acc",
                "num_epochs",
                "training_time",
            ]:
                params[key] = value

        return params

    def plot_results(self, output_dir: str) -> None:
        """Generate plots of grid search results."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            self.logger.warning("Matplotlib/Seaborn not available, skipping plots")
            return

        if not self.results:
            return

        successful_runs = [r for r in self.results if r.get("status") == "completed"]
        if not successful_runs:
            return

        # Create plots directory
        plots_dir = os.path.join(output_dir, "plots")
        ensure_directory(plots_dir)

        # Plot validation accuracy vs parameters
        self._plot_parameter_analysis(successful_runs, plots_dir)

        # Plot training curves for best runs
        self._plot_training_curves(successful_runs, plots_dir)

        self.logger.info(f"Plots saved to: {plots_dir}")

    def _plot_parameter_analysis(
        self, successful_runs: List[Dict], plots_dir: str
    ) -> None:
        """Plot parameter analysis."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        # This is a placeholder for parameter analysis plots
        # In a full implementation, you would create various plots showing
        # how different parameters affect performance

        plt.figure(figsize=(12, 8))
        plt.text(
            0.5,
            0.5,
            "Parameter analysis plots would be generated here",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
            fontsize=16,
        )
        plt.title("Grid Search Parameter Analysis")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(
            os.path.join(plots_dir, "parameter_analysis.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_training_curves(
        self, successful_runs: List[Dict], plots_dir: str
    ) -> None:
        """Plot training curves for best runs."""
        import matplotlib.pyplot as plt

        # This is a placeholder for training curve plots
        # In a full implementation, you would load and plot the training histories

        plt.figure(figsize=(12, 8))
        plt.text(
            0.5,
            0.5,
            "Training curve plots would be generated here",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
            fontsize=16,
        )
        plt.title("Grid Search Training Curves")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(
            os.path.join(plots_dir, "training_curves.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _create_param_hash(self, params: Dict[str, Any]) -> str:
        """Create a hash for parameter combination to track completion."""
        import hashlib
        # Sort parameters for consistent hashing
        sorted_params = sorted(params.items())
        param_str = str(sorted_params)
        return hashlib.md5(param_str.encode()).hexdigest()

    def _load_existing_results(self, base_output_dir: str) -> None:
        """Load existing results and completed combinations."""
        results_file = os.path.join(base_output_dir, "grid_search_results.json")
        status_file = os.path.join(base_output_dir, "grid_search_status.json")
        
        # Load existing results
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    self.results = json.load(f)
                # Filter results to only those whose output directories still exist
                filtered_results = []
                for r in self.results:
                    out_dir = r.get('output_dir')
                    if isinstance(out_dir, str) and os.path.exists(out_dir):
                        filtered_results.append(r)
                removed = len(self.results) - len(filtered_results)
                self.results = filtered_results
                if removed > 0:
                    self.logger.info(f"Pruned {removed} stale result entries with missing output directories")
                self.logger.info(f"Loaded {len(self.results)} existing results")
            except Exception as e:
                self.logger.warning(f"Could not load existing results: {e}")
                self.results = []
        
        # Load completed combinations from status file
        if os.path.exists(status_file):
            try:
                with open(status_file, 'r') as f:
                    status_data = json.load(f)
                    loaded_completed = set(status_data.get('completed_combinations', []))
                # Cross-check status completed combinations against actually present output dirs
                validated_completed = set()
                # Build a quick lookup of param_hash -> output_dir from results
                hash_to_dir = {r.get('param_hash'): r.get('output_dir') for r in self.results if r.get('param_hash')}
                for h in loaded_completed:
                    out_dir = hash_to_dir.get(h)
                    if out_dir and os.path.exists(out_dir):
                        validated_completed.add(h)
                self.completed_combinations = validated_completed
                self.logger.info(f"Loaded {len(self.completed_combinations)} completed combinations (validated)")
            except Exception as e:
                self.logger.warning(f"Could not load status file: {e}")
                self.completed_combinations = set()
        else:
            # Extract completed combinations from existing results
            self.completed_combinations = set()
            for result in self.results:
                if 'param_hash' in result:
                    out_dir = result.get('output_dir')
                    if isinstance(out_dir, str) and os.path.exists(out_dir):
                        self.completed_combinations.add(result['param_hash'])

    def _build_dir_name_from_params(self, params: Dict[str, Any]) -> str:
        """Build the directory name for a set of parameters without creating it."""
        dir_parts = []
        for param_name, param_value in params.items():
            if isinstance(param_value, float):
                param_str = f"{param_name}{param_value:.3f}".replace(".", "p")
            else:
                param_str = f"{param_name}{param_value}"
            dir_parts.append(param_str)
        return "_".join(dir_parts)

    def _save_checkpoint(self) -> None:
        """Save current progress as checkpoint."""
        if not self.status_file:
            return
            
        try:
            # Save current results
            results_file = os.path.join(os.path.dirname(self.status_file), "grid_search_results.json")
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            
            # Save status
            status_data = {
                'completed_combinations': list(self.completed_combinations),
                'total_results': len(self.results),
                'last_updated': time.time()
            }
            with open(self.status_file, 'w') as f:
                json.dump(status_data, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"Could not save checkpoint: {e}")

    def get_progress_summary(self) -> Dict[str, Any]:
        """Get summary of current progress."""
        total_results = len(self.results)
        successful = len([r for r in self.results if r.get('status') == 'completed'])
        failed = len([r for r in self.results if r.get('status') == 'failed'])
        
        return {
            'total_combinations': total_results,
            'completed_combinations': len(self.completed_combinations),
            'successful_runs': successful,
            'failed_runs': failed,
            'success_rate': successful / total_results if total_results > 0 else 0
        }

    def _generate_summary_report(self, output_dir: str, summary: Dict[str, Any]) -> None:
        """Generate a detailed text summary report."""
        report_file = os.path.join(output_dir, "grid_search_report.txt")
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("GRID SEARCH SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Overall statistics
            f.write("OVERALL STATISTICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total combinations tested: {summary.get('total_combinations', 0)}\n")
            f.write(f"Successful runs: {summary.get('successful_runs', 0)}\n")
            f.write(f"Failed runs: {summary.get('failed_runs', 0)}\n")
            f.write(f"Success rate: {summary.get('success_rate', 0)*100:.1f}%\n\n")
            
            if 'statistics' in summary:
                stats = summary['statistics']
                f.write("PERFORMANCE STATISTICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Validation Accuracy:\n")
                f.write(f"  Mean: {stats.get('mean_val_acc', 0):.4f}\n")
                f.write(f"  Std:  {stats.get('std_val_acc', 0):.4f}\n")
                f.write(f"  Min:  {stats.get('min_val_acc', 0):.4f}\n")
                f.write(f"  Max:  {stats.get('max_val_acc', 0):.4f}\n\n")
                
                f.write(f"Validation Loss:\n")
                f.write(f"  Mean: {stats.get('mean_val_loss', 0):.4f}\n")
                f.write(f"  Std:  {stats.get('std_val_loss', 0):.4f}\n")
                f.write(f"  Min:  {stats.get('min_val_loss', 0):.4f}\n")
                f.write(f"  Max:  {stats.get('max_val_loss', 0):.4f}\n\n")
            
            # Best configurations
            if 'best_accuracy' in summary:
                best_acc = summary['best_accuracy']
                f.write("BEST CONFIGURATION (Highest Accuracy):\n")
                f.write("-" * 40 + "\n")
                f.write(f"Validation Accuracy: {best_acc.get('value', 0):.4f}\n")
                f.write("Parameters:\n")
                for param, value in best_acc.get('parameters', {}).items():
                    f.write(f"  {param}: {value}\n")
                f.write("\n")
            
            if 'best_loss' in summary:
                best_loss = summary['best_loss']
                f.write("BEST CONFIGURATION (Lowest Loss):\n")
                f.write("-" * 40 + "\n")
                f.write(f"Validation Loss: {best_loss.get('value', 0):.4f}\n")
                f.write("Parameters:\n")
                for param, value in best_loss.get('parameters', {}).items():
                    f.write(f"  {param}: {value}\n")
                f.write("\n")
            
            f.write("FILES GENERATED:\n")
            f.write("-" * 40 + "\n")
            f.write("• grid_search_results.json - Complete results in JSON format\n")
            f.write("• grid_search_results.csv - Results in CSV format for analysis\n")
            f.write("• grid_search_status.json - Progress tracking information\n")
            f.write("• grid_search_summary.json - Statistical summary in JSON format\n")
            f.write("• grid_search_report.txt - This human-readable report\n")
            f.write("• plots/ - Visualization directory containing:\n")
            f.write("  - top_performers.png - Top 10 performing configurations\n")
            f.write("  - parameter_accuracy.png - Parameter vs accuracy scatter plots\n")
            f.write("  - performance_distribution.png - Accuracy and loss distributions\n")
            f.write("  - parameter_correlation.png - Parameter correlation heatmap\n")
            f.write("  - parameter_importance.png - Parameter importance ranking\n")
            f.write("• Individual model directories for each combination\n")
            
        self.logger.info(f"Summary report saved to {report_file}")

    def _generate_visualization_plots(self, output_dir: str, summary: Dict[str, Any]) -> None:
        """Generate visualization plots for grid search results."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import pandas as pd
        except ImportError:
            self.logger.warning("Matplotlib/Seaborn not available, skipping plots")
            return

        # Create plots directory
        plots_dir = os.path.join(output_dir, "plots")
        ensure_directory(plots_dir)

        # Convert results to DataFrame for easier plotting
        df = pd.DataFrame(self.results)
        successful_df = df[df['status'] == 'completed'].copy()

        if successful_df.empty:
            self.logger.warning("No successful runs to plot")
            return

        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")

        # 1. Top Performers Bar Chart
        self._plot_top_performers(successful_df, plots_dir)

        # 2. Parameter vs Accuracy Scatter Plots
        self._plot_parameter_accuracy(successful_df, plots_dir)

        # 3. Performance Distribution
        self._plot_performance_distribution(successful_df, plots_dir)

        # 4. Parameter Importance Heatmap
        self._plot_parameter_importance(successful_df, plots_dir)

        self.logger.info(f"Visualization plots saved to {plots_dir}")

    def _plot_top_performers(self, df: pd.DataFrame, plots_dir: str) -> None:
        """Plot top performing configurations."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Get top 10 performers using test accuracy
        top_10 = df.nlargest(10, 'test_acc')
        
        plt.figure(figsize=(12, 8))
        
        # Create a combined label for each configuration
        labels = []
        for _, row in top_10.iterrows():
            label_parts = []
            for col in ['hidden_size', 'num_layers', 'dropout', 'learning_rate', 'batch_size']:
                if col in row:
                    if col == 'learning_rate':
                        label_parts.append(f"{col}={row[col]:.0e}")
                    elif col == 'dropout':
                        label_parts.append(f"{col}={row[col]:.1f}")
                    else:
                        label_parts.append(f"{col}={row[col]}")
            labels.append('\n'.join(label_parts))
        
        bars = plt.bar(range(len(top_10)), top_10['test_acc'], 
                      color=sns.color_palette("husl", len(top_10)))
        
        plt.xlabel('Configuration')
        plt.ylabel('Test Accuracy')
        plt.title('Top 10 Performing Configurations', fontsize=16, fontweight='bold')
        plt.xticks(range(len(top_10)), labels, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (bar, acc) in enumerate(zip(bars, top_10['test_acc'])):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{acc:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'top_performers.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_parameter_accuracy(self, df: pd.DataFrame, plots_dir: str) -> None:
        """Plot parameter values vs accuracy."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Parameters to plot
        param_cols = ['hidden_size', 'num_layers', 'dropout', 'learning_rate', 'batch_size']
        param_cols = [col for col in param_cols if col in df.columns]
        
        if not param_cols:
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, param in enumerate(param_cols):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Create scatter plot
            scatter = ax.scatter(df[param], df['test_acc'], 
                               c=df['test_loss'], cmap='viridis', 
                               alpha=0.7, s=50)
            
            ax.set_xlabel(param.replace('_', ' ').title())
            ax.set_ylabel('Test Accuracy')
            ax.set_title(f'{param.replace("_", " ").title()} vs Accuracy')
            
            # Add colorbar for loss
            plt.colorbar(scatter, ax=ax, label='Test Loss')
            
            # Add trend line
            if df[param].dtype in ['int64', 'float64'] and len(df) > 1:
                try:
                    z = np.polyfit(df[param], df['test_acc'], 1)
                    p = np.poly1d(z)
                    ax.plot(df[param], p(df[param]), "r--", alpha=0.8)
                except (np.linalg.LinAlgError, np.RankWarning):
                    # Skip trend line if polyfit fails
                    pass
        
        # Remove empty subplots
        for i in range(len(param_cols), len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle('Parameter Values vs Model Performance', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'parameter_accuracy.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_performance_distribution(self, df: pd.DataFrame, plots_dir: str) -> None:
        """Plot performance distribution."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy distribution
        ax1.hist(df['test_acc'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(df['test_acc'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["test_acc"].mean():.4f}')
        ax1.axvline(df['test_acc'].median(), color='green', linestyle='--', 
                   label=f'Median: {df["test_acc"].median():.4f}')
        ax1.set_xlabel('Test Accuracy')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Test Accuracy')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Loss distribution
        ax2.hist(df['test_loss'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.axvline(df['test_loss'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["test_loss"].mean():.4f}')
        ax2.axvline(df['test_loss'].median(), color='green', linestyle='--', 
                   label=f'Median: {df["test_loss"].median():.4f}')
        ax2.set_xlabel('Test Loss')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Test Loss')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.suptitle('Performance Distribution Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'performance_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_parameter_importance(self, df: pd.DataFrame, plots_dir: str) -> None:
        """Plot parameter importance heatmap."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy.stats import spearmanr

        # Parameters to analyze
        param_cols = ['hidden_size', 'num_layers', 'dropout', 'learning_rate', 'batch_size', 'weight_decay']
        param_cols = [col for col in param_cols if col in df.columns]
        
        if len(param_cols) < 2 or len(df) < 2:
            self.logger.warning("Not enough parameters or data points for correlation analysis, skipping")
            return

        # Calculate correlation matrix
        corr_matrix = df[param_cols + ['test_acc']].corr()
        
        # Check for valid data
        if corr_matrix.isnull().all().all():
            self.logger.warning("All correlation values are NaN, skipping correlation heatmap")
            return
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Filter out NaN values for seaborn
        corr_matrix_clean = corr_matrix.fillna(0)
        
        sns.heatmap(corr_matrix_clean, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        plt.title('Parameter Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'parameter_correlation.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Parameter importance (correlation with accuracy)
        if 'test_acc' in corr_matrix.columns:
            importance = corr_matrix['test_acc'].drop('test_acc').abs().sort_values(ascending=True)
            
            # Only plot if we have valid importance values
            if not importance.empty and not importance.isnull().all():
                plt.figure(figsize=(10, 6))
                bars = plt.barh(range(len(importance)), importance.values, 
                               color=sns.color_palette("viridis", len(importance)))
                
                plt.yticks(range(len(importance)), importance.index)
                plt.xlabel('Absolute Correlation with Accuracy')
                plt.title('Parameter Importance (Correlation with Accuracy)', fontsize=16, fontweight='bold')
                
                # Add value labels
                for i, (bar, val) in enumerate(zip(bars, importance.values)):
                    if not np.isnan(val):
                        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                                f'{val:.3f}', ha='left', va='center', fontsize=10)
                
                plt.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'parameter_importance.png'), dpi=300, bbox_inches='tight')
                plt.close()
            else:
                self.logger.warning("No valid parameter importance data, skipping importance plot")
