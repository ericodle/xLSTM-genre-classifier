#!/usr/bin/env python3
"""
One-Factor-at-a-Time (OFAT) Analysis for GenreDiscern Models

This script performs systematic parameter sensitivity analysis by varying
one parameter at a time while holding all others constant.

Usage:
    python run_ofat_analysis.py --model CNN --data ./mfccs/gtzan_13.json --output ./output/ofat_analysis
"""

import argparse
import json
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from training.grid_search import GridSearchTrainer
    from core.config import Config
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)


class OFATAnalyzer:
    """One-Factor-at-a-Time parameter sensitivity analysis."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.results = {}
        self.baseline_config = None
        self.parameter_ranges = None
        self.parameters_to_test = None
        
    def load_ofat_config(self, model_type: str, config_file: str = None) -> Dict[str, Any]:
        """Load OFAT configuration from JSON file."""
        if config_file is None:
            config_file = f"ofat_configs/{model_type.lower()}_ofat_config.json"
        
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"OFAT config file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        return config
    
    def define_baseline_config(self, model_type: str, config_file: str = None) -> Dict[str, Any]:
        """Load baseline configuration from JSON file."""
        ofat_config = self.load_ofat_config(model_type, config_file)
        return ofat_config['baseline_config']
    
    def define_parameter_ranges(self, model_type: str, config_file: str = None) -> Dict[str, List[Any]]:
        """Load parameter ranges from JSON file."""
        ofat_config = self.load_ofat_config(model_type, config_file)
        return ofat_config['parameter_ranges']
    
    def get_parameters_to_test(self, model_type: str, config_file: str = None) -> List[str]:
        """Load parameters to test from JSON file."""
        ofat_config = self.load_ofat_config(model_type, config_file)
        return ofat_config.get('parameters_to_test', list(ofat_config['parameter_ranges'].keys()))
    
    def run_ofat_analysis(self, 
                         data_path: str, 
                         model_type: str, 
                         output_dir: str,
                         parameters_to_test: List[str] = None,
                         config_file: str = None) -> Dict[str, pd.DataFrame]:
        """
        Run OFAT analysis for specified parameters.
        
        Args:
            data_path: Path to the data file
            model_type: Type of model to analyze
            output_dir: Directory to save results
            parameters_to_test: List of parameters to test (None = use config file)
            config_file: Path to OFAT config file (None = use default)
        """
        print(f"🔬 Starting OFAT Analysis for {model_type} Model")
        print("=" * 60)
        
        # Load configuration from file
        self.baseline_config = self.define_baseline_config(model_type, config_file)
        parameter_ranges = self.define_parameter_ranges(model_type, config_file)
        
        # Determine which parameters to test
        if parameters_to_test is None:
            parameters_to_test = self.get_parameters_to_test(model_type, config_file)
        
        print(f"📊 Testing parameters: {parameters_to_test}")
        print(f"🎯 Baseline config: {self.baseline_config}")
        print()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Run OFAT for each parameter
        all_results = {}
        
        for param_name in parameters_to_test:
            if param_name not in parameter_ranges:
                print(f"⚠️  Skipping unknown parameter: {param_name}")
                continue
                
            print(f"🔍 Testing parameter: {param_name}")
            print(f"   Values: {parameter_ranges[param_name]}")
            
            # Create parameter grid for this parameter
            param_grid = {model_type: {}}
            for param, value in self.baseline_config.items():
                if param == param_name:
                    param_grid[model_type][param] = parameter_ranges[param_name]
                else:
                    param_grid[model_type][param] = [value]
            
            # Run grid search for this parameter
            try:
                grid_trainer = GridSearchTrainer()
                results = grid_trainer.run_grid_search(
                    data_path=data_path,
                    model_type=model_type,
                    base_output_dir=os.path.join(output_dir, f"ofat_{param_name}"),
                    param_grid=param_grid[model_type],
                    resume=True
                )
                
                # Store results
                all_results[param_name] = results
                print(f"   ✅ Completed {len(results)} runs")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                continue
        
        # Save and analyze results
        self.results = all_results
        self._save_results(output_dir)
        self._generate_analysis_plots(output_dir)
        self._generate_summary_report(output_dir)
        self._generate_optimal_config_suggestion(output_dir)
        
        return all_results
    
    def _save_results(self, output_dir: str):
        """Save OFAT results to files."""
        # Save individual parameter results
        for param_name, results in self.results.items():
            results_file = os.path.join(output_dir, f"{param_name}_ofat_results.csv")
            results.to_csv(results_file, index=False)
            print(f"💾 Saved {param_name} results to {results_file}")
        
        # Save combined results
        combined_results = []
        for param_name, results in self.results.items():
            results_copy = results.copy()
            results_copy['parameter_tested'] = param_name
            combined_results.append(results_copy)
        
        if combined_results:
            combined_df = pd.concat(combined_results, ignore_index=True)
            combined_file = os.path.join(output_dir, "ofat_combined_results.csv")
            combined_df.to_csv(combined_file, index=False)
            print(f"💾 Saved combined results to {combined_file}")
    
    def _generate_analysis_plots(self, output_dir: str):
        """Generate analysis plots for OFAT results."""
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Parameter sensitivity plot
        self._plot_parameter_sensitivity(plots_dir)
        
        # 2. Overfitting analysis plot
        self._plot_overfitting_analysis(plots_dir)
        
        # 3. Performance comparison plot
        self._plot_performance_comparison(plots_dir)
        
        
        print(f"📊 Generated analysis plots in {plots_dir}")
    
    def _plot_parameter_sensitivity(self, plots_dir: str):
        """Plot parameter sensitivity analysis."""
        param_names = list(self.results.keys())
        n_params = len(param_names)
        
        # Calculate grid dimensions - aim for roughly square layout
        n_cols = min(3, n_params)  # Max 3 columns
        n_rows = (n_params + n_cols - 1) // n_cols  # Calculate rows needed
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        # Handle single parameter case
        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        else:
            axes = axes.flatten()
        
        for i, param_name in enumerate(param_names):
            if i >= len(axes):
                break
                
            results = self.results[param_name]
            ax = axes[i]
            
            # Extract parameter values and accuracies
            param_values = results[param_name].values
            accuracies = results['test_acc'].values
            
            # Create line plot
            ax.plot(range(len(param_values)), accuracies, 'o-', linewidth=2, markersize=8)
            ax.set_xlabel(f'{param_name}')
            ax.set_ylabel('Test Accuracy')
            ax.set_title(f'Parameter Sensitivity: {param_name}')
            ax.grid(True, alpha=0.3)
            
            # Set x-axis labels
            ax.set_xticks(range(len(param_values)))
            ax.set_xticklabels([str(v) for v in param_values], rotation=45)
        
        # Hide empty subplots
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'parameter_sensitivity.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_overfitting_analysis(self, plots_dir: str):
        """Plot overfitting analysis showing train vs validation accuracy gap."""
        param_names = list(self.results.keys())
        n_params = len(param_names)
        
        # Calculate grid dimensions - aim for roughly square layout
        n_cols = min(3, n_params)  # Max 3 columns
        n_rows = (n_params + n_cols - 1) // n_cols  # Calculate rows needed
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        # Handle single parameter case
        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        else:
            axes = axes.flatten()
        
        for i, param_name in enumerate(param_names):
            if i >= len(axes):
                break
                
            results = self.results[param_name]
            ax = axes[i]
            
            # Extract parameter values and accuracies
            param_values = results[param_name].values
            
            # Check if we have the required columns
            if 'final_train_acc' not in results.columns or 'final_val_acc' not in results.columns:
                ax.text(0.5, 0.5, 'Overfitting data\nnot available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Overfitting Analysis: {param_name}')
                continue
            
            train_accs = results['final_train_acc'].values
            val_accs = results['final_val_acc'].values
            
            # Calculate overfitting gap (train - val)
            overfitting_gap = train_accs - val_accs
            
            # Create dual y-axis plot
            ax2 = ax.twinx()
            
            # Plot accuracies
            x_pos = range(len(param_values))
            line1 = ax.plot(x_pos, train_accs, 'o-', color='blue', linewidth=2, 
                           markersize=6, label='Train Acc', alpha=0.8)
            line2 = ax.plot(x_pos, val_accs, 's-', color='red', linewidth=2, 
                           markersize=6, label='Val Acc', alpha=0.8)
            
            # Plot overfitting gap as bars
            bars = ax2.bar(x_pos, overfitting_gap, alpha=0.3, color='orange', 
                          label='Overfitting Gap', width=0.6)
            
            # Color bars based on overfitting severity
            for j, gap in enumerate(overfitting_gap):
                if gap > 0.1:  # High overfitting
                    bars[j].set_color('red')
                elif gap > 0.05:  # Medium overfitting
                    bars[j].set_color('orange')
                else:  # Low overfitting
                    bars[j].set_color('green')
            
            # Labels and formatting
            ax.set_xlabel(f'{param_name}')
            ax.set_ylabel('Accuracy', color='black')
            ax2.set_ylabel('Overfitting Gap (Train - Val)', color='orange')
            ax.set_title(f'Overfitting Analysis: {param_name}')
            
            # Set x-axis labels
            ax.set_xticks(x_pos)
            ax.set_xticklabels([str(v) for v in param_values], rotation=45)
            
            # Grid
            ax.grid(True, alpha=0.3)
            
            # Position legends to avoid overlap
            # Main legend (accuracy lines) - top right
            lines1, labels1 = ax.get_legend_handles_labels()
            legend1 = ax.legend(lines1, labels1, loc='upper right', fontsize=8, 
                              framealpha=0.95, fancybox=True, shadow=True,
                              borderpad=0.5, columnspacing=0.5)
            legend1.set_zorder(1000)  # Ensure it's on top
            
            # Secondary legend (overfitting bars) - bottom right
            lines2, labels2 = ax2.get_legend_handles_labels()
            legend2 = ax2.legend(lines2, labels2, loc='lower right', fontsize=8,
                               framealpha=0.95, fancybox=True, shadow=True,
                               borderpad=0.5, columnspacing=0.5)
            legend2.set_zorder(1000)  # Ensure it's on top
            
            # Add overfitting statistics as text - positioned to avoid legends
            mean_gap = np.mean(overfitting_gap)
            max_gap = np.max(overfitting_gap)
            ax.text(0.02, 0.12, f'Mean Gap: {mean_gap:.3f}\nMax Gap: {max_gap:.3f}', 
                   transform=ax.transAxes, verticalalignment='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', 
                            alpha=0.9, edgecolor='navy', linewidth=0.5),
                   fontsize=7, family='monospace')
        
        # Hide empty subplots
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)
        
        # Adjust layout to prevent legend overlap
        plt.tight_layout(pad=2.0)
        
        # Additional padding for legends
        plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.9)
        
        plt.savefig(os.path.join(plots_dir, 'overfitting_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_comparison(self, plots_dir: str):
        """Plot performance comparison across parameters."""
        # Calculate parameter importance metrics
        param_importance = {}
        
        for param_name, results in self.results.items():
            accuracies = results['test_acc'].values
            param_importance[param_name] = {
                'range': np.max(accuracies) - np.min(accuracies),
                'std': np.std(accuracies),
                'max': np.max(accuracies),
                'min': np.min(accuracies)
            }
        
        # Create importance plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Range plot
        params = list(param_importance.keys())
        ranges = [param_importance[p]['range'] for p in params]
        ax1.bar(params, ranges, color='skyblue', alpha=0.7)
        ax1.set_title('Parameter Impact (Range)')
        ax1.set_ylabel('Accuracy Range')
        ax1.tick_params(axis='x', rotation=45)
        
        # Standard deviation plot
        stds = [param_importance[p]['std'] for p in params]
        ax2.bar(params, stds, color='lightcoral', alpha=0.7)
        ax2.set_title('Parameter Variability (Std Dev)')
        ax2.set_ylabel('Accuracy Std Dev')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'parameter_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_summary_report(self, output_dir: str):
        """Generate summary report of OFAT analysis."""
        report_file = os.path.join(output_dir, "ofat_summary_report.txt")
        
        with open(report_file, 'w') as f:
            f.write("🔬 OFAT (One-Factor-at-a-Time) Analysis Summary\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Baseline Configuration:\n")
            for param, value in self.baseline_config.items():
                f.write(f"  {param}: {value}\n")
            f.write("\n")
            
            f.write("Parameter Analysis Results:\n")
            f.write("-" * 40 + "\n")
            
            for param_name, results in self.results.items():
                accuracies = results['test_acc'].values
                param_values = results[param_name].values
                
                f.write(f"\n{param_name}:\n")
                f.write(f"  Range: {np.min(accuracies):.4f} - {np.max(accuracies):.4f}\n")
                f.write(f"  Impact: {np.max(accuracies) - np.min(accuracies):.4f}\n")
                f.write(f"  Std Dev: {np.std(accuracies):.4f}\n")
                f.write(f"  Best Value: {param_values[np.argmax(accuracies)]} (test_acc: {np.max(accuracies):.4f})\n")
                f.write(f"  Worst Value: {param_values[np.argmin(accuracies)]} (test_acc: {np.min(accuracies):.4f})\n")
                
                # Add overfitting analysis if data is available
                if 'final_train_acc' in results.columns and 'final_val_acc' in results.columns:
                    train_accs = results['final_train_acc'].values
                    val_accs = results['final_val_acc'].values
                    overfitting_gaps = train_accs - val_accs
                    
                    f.write(f"  Overfitting Analysis:\n")
                    f.write(f"    Mean Gap (Train-Val): {np.mean(overfitting_gaps):.4f}\n")
                    f.write(f"    Max Gap: {np.max(overfitting_gaps):.4f}\n")
                    f.write(f"    Min Gap: {np.min(overfitting_gaps):.4f}\n")
                    f.write(f"    Std Gap: {np.std(overfitting_gaps):.4f}\n")
                    
                    # Count overfitting severity
                    high_overfitting = np.sum(overfitting_gaps > 0.1)
                    medium_overfitting = np.sum((overfitting_gaps > 0.05) & (overfitting_gaps <= 0.1))
                    low_overfitting = np.sum(overfitting_gaps <= 0.05)
                    
                    f.write(f"    Overfitting Severity:\n")
                    f.write(f"      High (>10%): {high_overfitting}/{len(overfitting_gaps)}\n")
                    f.write(f"      Medium (5-10%): {medium_overfitting}/{len(overfitting_gaps)}\n")
                    f.write(f"      Low (<5%): {low_overfitting}/{len(overfitting_gaps)}\n")
            
            # Rank parameters by impact
            f.write("\nParameter Ranking by Impact:\n")
            f.write("-" * 40 + "\n")
            
            param_impacts = {}
            for param_name, results in self.results.items():
                accuracies = results['test_acc'].values
                param_impacts[param_name] = np.max(accuracies) - np.min(accuracies)
            
            sorted_params = sorted(param_impacts.items(), key=lambda x: x[1], reverse=True)
            for i, (param_name, impact) in enumerate(sorted_params, 1):
                f.write(f"{i:2d}. {param_name}: {impact:.4f}\n")
            
            # Rank parameters by overfitting tendency
            f.write("\nParameter Ranking by Overfitting Tendency:\n")
            f.write("-" * 40 + "\n")
            
            param_overfitting = {}
            for param_name, results in self.results.items():
                if 'final_train_acc' in results.columns and 'final_val_acc' in results.columns:
                    train_accs = results['final_train_acc'].values
                    val_accs = results['final_val_acc'].values
                    overfitting_gaps = train_accs - val_accs
                    param_overfitting[param_name] = np.mean(overfitting_gaps)
            
            if param_overfitting:
                sorted_overfitting = sorted(param_overfitting.items(), key=lambda x: x[1], reverse=True)
                for i, (param_name, avg_gap) in enumerate(sorted_overfitting, 1):
                    f.write(f"{i:2d}. {param_name}: {avg_gap:.4f} avg gap\n")
            else:
                f.write("No overfitting data available\n")
        
        print(f"📄 Generated summary report: {report_file}")
    
    def _generate_optimal_config_suggestion(self, output_dir: str):
        """Generate optimal hyperparameter configuration suggestion based on OFAT results."""
        suggestion_file = os.path.join(output_dir, "optimal_config_suggestion.json")
        
        if not self.results:
            print("⚠️  No OFAT results available for optimal configuration suggestion")
            return
        
        # Analyze each parameter to find optimal values
        optimal_config = {}
        confidence_scores = {}
        analysis_details = {}
        
        for param_name, results in self.results.items():
            # Get parameter values and test accuracies
            if param_name not in results.columns:
                print(f"⚠️  Parameter {param_name} not found in results, skipping...")
                continue
                
            param_values = results[param_name].values
            test_accs = results['test_acc'].values
            
            # Find the parameter value with highest test accuracy
            best_idx = np.argmax(test_accs)
            optimal_value = param_values[best_idx]
            best_accuracy = test_accs[best_idx]
            
            # Calculate confidence score based on performance gap and stability
            accuracy_range = np.max(test_accs) - np.min(test_accs)
            accuracy_std = np.std(test_accs)
            
            # Confidence based on how much better the best value is
            if accuracy_range > 0:
                confidence = min(1.0, (best_accuracy - np.mean(test_accs)) / accuracy_range)
            else:
                confidence = 0.5  # Neutral confidence if no variation
            
            # Check for overfitting if data is available
            overfitting_penalty = 0
            if 'final_train_acc' in results.columns and 'final_val_acc' in results.columns:
                train_accs = results['final_train_acc'].values
                val_accs = results['final_val_acc'].values
                overfitting_gaps = train_accs - val_accs
                
                # Penalize high overfitting
                best_overfitting_gap = overfitting_gaps[best_idx]
                if best_overfitting_gap > 0.1:  # High overfitting
                    overfitting_penalty = 0.2
                elif best_overfitting_gap > 0.05:  # Medium overfitting
                    overfitting_penalty = 0.1
                
                confidence = max(0.1, confidence - overfitting_penalty)
            
            # Store results (convert numpy types to Python native types)
            optimal_config[param_name] = int(optimal_value) if isinstance(optimal_value, np.integer) else float(optimal_value)
            confidence_scores[param_name] = float(confidence)
            
            analysis_details[param_name] = {
                "optimal_value": int(optimal_value) if isinstance(optimal_value, np.integer) else float(optimal_value),
                "test_accuracy": float(best_accuracy),
                "accuracy_range": float(accuracy_range),
                "accuracy_std": float(accuracy_std),
                "confidence_score": float(confidence),
                "overfitting_gap": float(overfitting_gaps[best_idx]) if 'final_train_acc' in results.columns else None,
                "all_values": [int(v) if isinstance(v, np.integer) else float(v) for v in param_values.tolist()],
                "all_accuracies": test_accs.tolist()
            }
        
        # Calculate overall confidence
        overall_confidence = np.mean(list(confidence_scores.values()))
        
        # Generate recommendation
        recommendation = {
            "model_type": "Unknown",  # Will be filled by caller
            "optimal_configuration": optimal_config,
            "confidence_scores": confidence_scores,
            "overall_confidence": float(overall_confidence),
            "analysis_details": analysis_details,
            "recommendation_notes": self._generate_recommendation_notes(analysis_details, overall_confidence),
            "usage_instructions": {
                "description": "This configuration was derived from OFAT analysis",
                "confidence": f"{overall_confidence:.1%}",
                "note": "Test this configuration and consider fine-tuning based on results"
            }
        }
        
        # Save to file
        try:
            with open(suggestion_file, 'w') as f:
                json.dump(recommendation, f, indent=2)
        except Exception as e:
            print(f"❌ Error saving optimal config suggestion: {e}")
            return
        
        # Print summary
        print(f"🎯 Optimal Configuration Suggestion:")
        print(f"   Overall Confidence: {overall_confidence:.1%}")
        print(f"   Configuration: {optimal_config}")
        print(f"📄 Saved to: {suggestion_file}")
    
    def _generate_recommendation_notes(self, analysis_details: dict, overall_confidence: float) -> list:
        """Generate human-readable recommendation notes."""
        notes = []
        
        # Overall confidence note
        if overall_confidence > 0.8:
            notes.append("High confidence: OFAT results show clear parameter preferences")
        elif overall_confidence > 0.6:
            notes.append("Medium confidence: Some parameters show clear preferences, others are less certain")
        else:
            notes.append("Low confidence: Parameter effects are subtle, consider more extensive search")
        
        # Parameter-specific notes
        high_impact_params = []
        low_impact_params = []
        overfitting_params = []
        
        for param_name, details in analysis_details.items():
            if details["accuracy_range"] > 0.05:  # High impact
                high_impact_params.append(param_name)
            elif details["accuracy_range"] < 0.01:  # Low impact
                low_impact_params.append(param_name)
            
            if details["overfitting_gap"] and details["overfitting_gap"] > 0.1:
                overfitting_params.append(param_name)
        
        if high_impact_params:
            notes.append(f"High-impact parameters: {', '.join(high_impact_params)} - these significantly affect performance")
        
        if low_impact_params:
            notes.append(f"Low-impact parameters: {', '.join(low_impact_params)} - these have minimal effect on performance")
        
        if overfitting_params:
            notes.append(f"Overfitting-prone parameters: {', '.join(overfitting_params)} - monitor train/val gap when using these values")
        
        # General recommendations
        if overall_confidence < 0.7:
            notes.append("Consider running grid search on high-impact parameters for better optimization")
        
        notes.append("Test this configuration and compare with baseline performance")
        notes.append("Consider ensemble methods if individual parameter effects are unclear")
        
        return notes


def setup_parser():
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Run OFAT (One-Factor-at-a-Time) parameter sensitivity analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--model', '-m',
        required=True,
        choices=['CNN', 'FC', 'LSTM', 'GRU', 'Transformer', 'xLSTM'],
        help='Model type to analyze'
    )
    
    parser.add_argument(
        '--data', '-d',
        required=True,
        help='Path to MFCC data file (JSON format)'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output directory for OFAT analysis results'
    )
    
    parser.add_argument(
        '--params', '-p',
        nargs='+',
        help='Specific parameters to test (default: use config file)'
    )
    
    parser.add_argument(
        '--config', '-c',
        help='Path to custom OFAT config file (default: ofat_configs/{model}_ofat_config.json)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for training (default: 16, smaller = less memory)'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=10000,
        help='Maximum number of samples to use (default: 10000 for memory-constrained systems)'
    )
    
    parser.add_argument(
        '--memory-efficient',
        action='store_true',
        help='Use memory-efficient data loading (loads data in chunks)'
    )
    
    return parser


def main():
    """Main OFAT analysis execution."""
    parser = setup_parser()
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.data):
        print(f"Error: Data file not found: {args.data}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Run OFAT analysis
    print(f"🎯 OFAT Analysis for {args.model} Model")
    print(f"📁 Data: {args.data}")
    print(f"📂 Output: {args.output}")
    if args.params:
        print(f"🔧 Parameters: {args.params}")
    else:
        print(f"🔧 Parameters: All available")
    print()
    
    try:
        analyzer = OFATAnalyzer()
        results = analyzer.run_ofat_analysis(
            data_path=args.data,
            model_type=args.model,
            output_dir=args.output,
            parameters_to_test=args.params,
            config_file=args.config
        )
        
        print(f"\n" + "="*60)
        print(f"🎯 OFAT ANALYSIS COMPLETED!")
        print(f"="*60)
        print(f"📊 Analyzed {len(results)} parameters")
        print(f"📁 Results saved to: {args.output}")
        print(f"📈 Plots saved to: {os.path.join(args.output, 'plots')}")
        print(f"📄 Summary report: {os.path.join(args.output, 'ofat_summary_report.txt')}")
        
    except Exception as e:
        print(f"❌ OFAT analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
