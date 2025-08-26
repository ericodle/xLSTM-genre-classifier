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


class GridSearchTrainer:
    """Grid search hyperparameter optimization."""
    
    def __init__(self, config: Optional[Config] = None, logger: Optional[logging.Logger] = None):
        self.config = config or Config()
        self.logger = logger or setup_logging()
        self.results = []
        
    def run_grid_search(
        self,
        data_path: str,
        model_type: str,
        base_output_dir: str,
        param_grid: Dict[str, List[Any]],
        **kwargs
    ) -> pd.DataFrame:
        """
        Run grid search over hyperparameters.
        
        Args:
            data_path: Path to the data file
            model_type: Type of model to train
            base_output_dir: Base directory for output
            param_grid: Dictionary of parameter names to lists of values
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
        
        # Run training for each combination
        for i, param_values in enumerate(param_combinations):
            self.logger.info(f"Training combination {i+1}/{total_combinations}")
            
            # Create parameter dictionary
            params = dict(zip(param_names, param_values))
            
            # Create output directory for this combination
            output_dir = self._create_output_dir(base_output_dir, params)
            
            try:
                # Train model with these parameters
                result = self._train_single_combination(
                    data_path, model_type, output_dir, params, **kwargs
                )
                
                # Store results
                result.update(params)
                result['combination_id'] = i
                result['output_dir'] = output_dir
                self.results.append(result)
                
                self.logger.info(f"Combination {i+1} completed successfully")
                
            except Exception as e:
                self.logger.error(f"Combination {i+1} failed: {e}")
                # Store failure result
                failure_result = {
                    'combination_id': i,
                    'output_dir': output_dir,
                    'status': 'failed',
                    'error': str(e)
                }
                failure_result.update(params)
                self.results.append(failure_result)
        
        # Save results
        self._save_results(base_output_dir)
        
        # Generate summary
        summary = self._generate_summary()
        
        self.logger.info("Grid search completed")
        self.logger.info(f"Results saved to: {base_output_dir}")
        
        return pd.DataFrame(self.results)
    
    def _train_single_combination(
        self,
        data_path: str,
        model_type: str,
        output_dir: str,
        params: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Train a single parameter combination."""
        # Create trainer
        trainer = ModelTrainer(self.config, self.logger)
        
        # Override config with grid search parameters
        config_override = self.config
        for param_name, param_value in params.items():
            if hasattr(config_override.model, param_name):
                setattr(config_override.model, param_name, param_value)
            elif hasattr(config_override.training, param_name):
                setattr(config_override.training, param_name, param_value)
        
        # Setup training
        trainer.setup_training(
            data_path=data_path,
            model_type=model_type,
            output_dir=output_dir,
            **kwargs
        )
        
        # Train model
        training_history = trainer.train()
        
        # Get final metrics
        final_train_loss = training_history['train_losses'][-1] if training_history['train_losses'] else float('inf')
        final_val_loss = training_history['val_losses'][-1] if training_history['val_losses'] else float('inf')
        final_train_acc = training_history['train_accuracies'][-1] if training_history['train_accuracies'] else 0.0
        final_val_acc = training_history['val_accuracies'][-1] if training_history['val_accuracies'] else 0.0
        
        # Get best validation metrics
        best_val_loss = min(training_history['val_losses']) if training_history['val_losses'] else float('inf')
        best_val_acc = max(training_history['val_accuracies']) if training_history['val_accuracies'] else 0.0
        
        return {
            'status': 'completed',
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'final_train_acc': final_train_acc,
            'final_val_acc': final_val_acc,
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc,
            'num_epochs': len(training_history['train_losses']),
            'training_time': time.time()  # Placeholder for actual training time
        }
    
    def _create_output_dir(self, base_dir: str, params: Dict[str, Any]) -> str:
        """Create output directory name based on parameters."""
        # Create a readable directory name
        dir_parts = []
        for param_name, param_value in params.items():
            # Convert parameter value to string, handling special cases
            if isinstance(param_value, float):
                param_str = f"{param_name}{param_value:.3f}".replace('.', 'p')
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
        with open(results_file, 'w') as f:
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
        successful_runs = [r for r in self.results if r.get('status') == 'completed']
        failed_runs = [r for r in self.results if r.get('status') == 'failed']
        
        if not successful_runs:
            return {'status': 'all_failed', 'failed_count': len(failed_runs)}
        
        # Find best parameters
        best_val_acc_idx = np.argmax([r.get('best_val_acc', 0) for r in successful_runs])
        best_val_loss_idx = np.argmin([r.get('best_val_loss', float('inf')) for r in successful_runs])
        
        best_acc_run = successful_runs[best_val_acc_idx]
        best_loss_run = successful_runs[best_val_loss_idx]
        
        # Calculate statistics
        val_accuracies = [r.get('best_val_acc', 0) for r in successful_runs]
        val_losses = [r.get('best_val_loss', float('inf')) for r in successful_runs]
        
        summary = {
            'total_combinations': len(self.results),
            'successful_runs': len(successful_runs),
            'failed_runs': len(failed_runs),
            'success_rate': len(successful_runs) / len(self.results),
            'best_accuracy': {
                'value': best_acc_run.get('best_val_acc', 0),
                'parameters': {k: v for k, v in best_acc_run.items() 
                             if k not in ['status', 'combination_id', 'output_dir', 'error']}
            },
            'best_loss': {
                'value': best_loss_run.get('best_val_loss', float('inf')),
                'parameters': {k: v for k, v in best_loss_run.items() 
                             if k not in ['status', 'combination_id', 'output_dir', 'error']}
            },
            'statistics': {
                'mean_val_acc': np.mean(val_accuracies),
                'std_val_acc': np.std(val_accuracies),
                'mean_val_loss': np.mean(val_losses),
                'std_val_loss': np.std(val_losses)
            }
        }
        
        return summary
    
    def get_best_parameters(self, metric: str = 'accuracy') -> Optional[Dict[str, Any]]:
        """
        Get the best parameters based on a metric.
        
        Args:
            metric: Metric to optimize ('accuracy' or 'loss')
            
        Returns:
            Dictionary of best parameters
        """
        if not self.results:
            return None
        
        successful_runs = [r for r in self.results if r.get('status') == 'completed']
        if not successful_runs:
            return None
        
        if metric == 'accuracy':
            best_run = max(successful_runs, key=lambda x: x.get('best_val_acc', 0))
        elif metric == 'loss':
            best_run = min(successful_runs, key=lambda x: x.get('best_val_loss', float('inf')))
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Extract only parameter values
        params = {}
        for key, value in best_run.items():
            if key not in ['status', 'combination_id', 'output_dir', 'error', 
                          'final_train_loss', 'final_val_loss', 'final_train_acc', 
                          'final_val_acc', 'best_val_loss', 'best_val_acc', 
                          'num_epochs', 'training_time']:
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
        
        successful_runs = [r for r in self.results if r.get('status') == 'completed']
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
    
    def _plot_parameter_analysis(self, successful_runs: List[Dict], plots_dir: str) -> None:
        """Plot parameter analysis."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # This is a placeholder for parameter analysis plots
        # In a full implementation, you would create various plots showing
        # how different parameters affect performance
        
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.5, "Parameter analysis plots would be generated here", 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=16)
        plt.title('Grid Search Parameter Analysis')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'parameter_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_training_curves(self, successful_runs: List[Dict], plots_dir: str) -> None:
        """Plot training curves for best runs."""
        import matplotlib.pyplot as plt
        
        # This is a placeholder for training curve plots
        # In a full implementation, you would load and plot the training histories
        
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.5, "Training curve plots would be generated here", 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=16)
        plt.title('Grid Search Training Curves')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close() 