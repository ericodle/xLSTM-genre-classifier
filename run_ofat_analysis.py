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
        
        # 2. Performance comparison plot
        self._plot_performance_comparison(plots_dir)
        
        # 3. Statistical significance plot
        self._plot_statistical_significance(plots_dir)
        
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
    
    def _plot_statistical_significance(self, plots_dir: str):
        """Plot statistical significance analysis with proper statistical tests."""
        from scipy import stats
        
        # Calculate statistical significance for each parameter
        param_stats = {}
        
        for param_name, results in self.results.items():
            param_values = results[param_name].values
            accuracies = results['test_acc'].values
            
            # Get unique parameter values
            unique_values = np.unique(param_values)
            
            if len(unique_values) < 2:
                continue  # Skip if only one value
                
            # Group accuracies by parameter value
            groups = [accuracies[param_values == val] for val in unique_values]
            
            # Perform ANOVA test
            try:
                f_stat, p_value = stats.f_oneway(*groups)
                
                # Calculate effect size (eta squared)
                ss_between = sum(len(group) * (np.mean(group) - np.mean(accuracies))**2 for group in groups)
                ss_total = sum((acc - np.mean(accuracies))**2 for acc in accuracies)
                eta_squared = ss_between / ss_total if ss_total > 0 else 0
                
                # Calculate confidence intervals for each group
                confidence_intervals = []
                for group in groups:
                    if len(group) > 1:
                        mean = np.mean(group)
                        sem = stats.sem(group)
                        ci = stats.t.interval(0.95, len(group)-1, loc=mean, scale=sem)
                        confidence_intervals.append(ci)
                    else:
                        confidence_intervals.append((group[0], group[0]))
                
                param_stats[param_name] = {
                    'f_stat': f_stat,
                    'p_value': p_value,
                    'eta_squared': eta_squared,
                    'unique_values': unique_values,
                    'groups': groups,
                    'confidence_intervals': confidence_intervals
                }
            except Exception as e:
                print(f"Warning: Could not perform statistical test for {param_name}: {e}")
                continue
        
        if not param_stats:
            print("Warning: No parameters suitable for statistical analysis")
            return
        
        # Create plots
        n_params = len(param_stats)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (param_name, stats_data) in enumerate(param_stats.items()):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            unique_values = stats_data['unique_values']
            groups = stats_data['groups']
            confidence_intervals = stats_data['confidence_intervals']
            p_value = stats_data['p_value']
            eta_squared = stats_data['eta_squared']
            
            # Plot means with confidence intervals
            means = [np.mean(group) for group in groups]
            errors = [ci[1] - mean for ci, mean in zip(confidence_intervals, means)]
            
            bars = ax.bar(range(len(unique_values)), means, yerr=errors, 
                         capsize=5, alpha=0.7, color='skyblue', edgecolor='navy')
            
            # Color bars based on significance
            for j, bar in enumerate(bars):
                if p_value < 0.05:
                    bar.set_color('lightcoral' if means[j] < np.mean(means) else 'lightgreen')
                else:
                    bar.set_color('lightgray')
            
            ax.set_xlabel(f'{param_name}')
            ax.set_ylabel('Test Accuracy')
            ax.set_xticks(range(len(unique_values)))
            ax.set_xticklabels([str(v) for v in unique_values])
            ax.grid(True, alpha=0.3)
            
            # Add statistical information to title
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            ax.set_title(f'{param_name}\np={p_value:.3f} {significance}, η²={eta_squared:.3f}')
        
        # Hide empty subplots
        for i in range(n_params, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        # Add legend
        fig.legend(['Significant (p<0.05)', 'Not Significant'], loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'statistical_significance.png'), dpi=300, bbox_inches='tight')
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
        
        print(f"📄 Generated summary report: {report_file}")


def setup_parser():
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Run OFAT (One-Factor-at-a-Time) parameter sensitivity analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run OFAT analysis for CNN model (uses ofat_configs/cnn_ofat_config.json)
    python run_ofat_analysis.py --model CNN --data ./mfccs/gtzan_13.json --output ./output/ofat_analysis
    
    # Run OFAT for specific parameters only
    python run_ofat_analysis.py --model CNN --data ./mfccs/gtzan_13.json --output ./output/ofat_analysis --params conv_layers kernel_size dropout
    
    # Run OFAT with custom config file
    python run_ofat_analysis.py --model CNN --data ./mfccs/gtzan_13.json --output ./output/ofat_analysis --config my_custom_ofat_config.json
    
    # Run OFAT for FC model
    python run_ofat_analysis.py --model FC --data ./mfccs/gtzan_13.json --output ./output/ofat_analysis
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        required=True,
        choices=['CNN', 'FC', 'LSTM', 'GRU'],
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
