#!/usr/bin/env python3
"""
Grid Search Runner for GenreDiscern Models

Usage:
    python run_grid_search.py --model GRU --data ./mfccs/gtzan_mfcc.json --output ./output/gru_gridsearch
    python run_grid_search.py --model LSTM --data ./mfccs/gtzan_mfcc.json --output ./output/lstm_gridsearch --params gru_params.json
"""

import argparse
import json
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from training.grid_search import GridSearchTrainer
    from core.config import Config
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)


def load_param_grid(params_file: str = None) -> dict:
    """Load parameter grid from file or use defaults."""
    if params_file:
        # If no path specified, look in training directory
        if not os.path.dirname(params_file):
            params_file = os.path.join('src', 'training', params_file)
        
        if os.path.exists(params_file):
            with open(params_file, 'r') as f:
                return json.load(f)
        else:
            print(f"Warning: Parameter file {params_file} not found, using defaults")
    
    # Default parameter grids for each model type
    default_grids = {
        'GRU': {
            'hidden_size': [32, 64, 128],
            'num_layers': [1, 2],
            'dropout': [0.1, 0.2, 0.3],
            'learning_rate': [0.001, 0.01],
            'batch_size': [16, 32]
        },
        'LSTM': {
            'hidden_size': [32, 64, 128],
            'num_layers': [1, 2],
            'dropout': [0.1, 0.2, 0.3],
            'learning_rate': [0.001, 0.01],
            'batch_size': [16, 32]
        },
        'CNN': {
            'num_filters': [32, 64],
            'kernel_size': [3, 5],
            'dropout': [0.1, 0.2],
            'learning_rate': [0.001, 0.01],
            'batch_size': [16, 32]
        },
        'FC': {
            'hidden_sizes': [[128, 64], [256, 128, 64]],
            'dropout': [0.1, 0.2],
            'learning_rate': [0.001, 0.01],
            'batch_size': [16, 32]
        },
        'xLSTM': {
            'hidden_size': [32, 64],
            'num_layers': [1, 2],
            'dropout': [0.1, 0.2],
            'learning_rate': [0.001, 0.01],
            'batch_size': [16, 32]
        }
    }
    
    return default_grids


def setup_parser():
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Run hyperparameter grid search for GenreDiscern models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run GRU grid search with default parameters (resume enabled by default)
    python run_grid_search.py --model GRU --data ./mfccs/gtzan_mfcc.json --output ./output/gru_gridsearch
    
    # Run LSTM grid search with custom parameters (will resume if interrupted)
    python run_grid_search.py --model LSTM --data ./mfccs/gtzan_mfcc.json --output ./output/lstm_gridsearch --params lstm_params.json
    
    # Start fresh (disable resume)
    python run_grid_search.py --model GRU --data ./mfccs/gtzan_mfcc.json --output ./output/gru_gridsearch --no-resume
    
    # Run with verbose logging
    python run_grid_search.py --model CNN --data ./mfccs/gtzan_mfcc.json --output ./output/cnn_gridsearch --verbose
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        required=True,
        choices=['GRU', 'LSTM', 'CNN', 'FC', 'xLSTM'],
        help='Model type to run grid search on'
    )
    
    parser.add_argument(
        '--data', '-d',
        required=True,
        help='Path to MFCC data file (JSON format)'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Base output directory for grid search results'
    )
    
    parser.add_argument(
        '--params', '-p',
        help='JSON file with custom parameter grid (optional)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be run without executing training'
    )
    
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Start fresh, do not resume from existing results (default: resume enabled)'
    )
    
    return parser


def main():
    """Main grid search execution."""
    parser = setup_parser()
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.data):
        print(f"Error: Data file not found: {args.data}")
        sys.exit(1)
    
    # Load parameter grid
    param_grid = load_param_grid(args.params)
    
    if args.model not in param_grid:
        print(f"Error: No default parameter grid for model type: {args.model}")
        sys.exit(1)
    
    # Get model-specific grid
    model_grid = param_grid[args.model]
    
    # Calculate total combinations
    total_combinations = 1
    for param_values in model_grid.values():
        total_combinations *= len(param_values)
    
    print(f"🎯 Grid Search for {args.model} Model")
    print(f"📁 Data: {args.data}")
    print(f"📂 Output: {args.output}")
    print(f"🔧 Parameters: {list(model_grid.keys())}")
    print(f"📊 Total combinations: {total_combinations}")
    print()
    
    if args.dry_run:
        print("🔍 DRY RUN - Parameter combinations that would be tested:")
        print()
        # Show first few combinations as example
        import itertools
        param_names = list(model_grid.keys())
        param_values = list(model_grid.values())
        
        for i, combination in enumerate(itertools.product(*param_values)):
            if i >= 5:  # Show only first 5
                print(f"... and {total_combinations - 5} more combinations")
                break
            params = dict(zip(param_names, combination))
            print(f"  {i+1:2d}. {params}")
        
        print(f"\nTotal training runs: {total_combinations}")
        return
    
    # Confirm before proceeding (skip in non-interactive mode)
    import sys
    if sys.stdin.isatty():
        print(f"This will run {total_combinations} training sessions.")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            print("Grid search cancelled.")
            return
    else:
        print(f"Running {total_combinations} training sessions in non-interactive mode...")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Save parameter grid for reference
    grid_info = {
        'model_type': args.model,
        'data_path': args.data,
        'parameter_grid': model_grid,
        'total_combinations': total_combinations
    }
    
    with open(os.path.join(args.output, 'grid_search_config.json'), 'w') as f:
        json.dump(grid_info, f, indent=2)
    
    # Run grid search
    print(f"\n🚀 Starting grid search...")
    try:
        grid_trainer = GridSearchTrainer()
        results = grid_trainer.run_grid_search(
            data_path=args.data,
            model_type=args.model,
            base_output_dir=args.output,
            param_grid=model_grid,
            resume=not args.no_resume
        )
        
        # Display comprehensive results summary
        successful_results = results[results['status'] == 'completed']
        failed_results = results[results['status'] == 'failed']
        
        print(f"\n" + "="*60)
        print(f"🎯 GRID SEARCH COMPLETED!")
        print(f"="*60)
        
        # Basic statistics
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   ✅ Successful runs: {len(successful_results)}")
        print(f"   ❌ Failed runs: {len(failed_results)}")
        print(f"   📈 Success rate: {len(successful_results)/len(results)*100:.1f}%")
        
        if not successful_results.empty:
            # Performance statistics
            val_accs = successful_results['best_val_acc'].values
            val_losses = successful_results['best_val_loss'].values
            
            print(f"\n🎯 PERFORMANCE STATISTICS:")
            print(f"   📊 Validation Accuracy:")
            print(f"      • Best:  {val_accs.max():.4f}")
            print(f"      • Mean:  {val_accs.mean():.4f}")
            print(f"      • Std:   {val_accs.std():.4f}")
            print(f"      • Min:   {val_accs.min():.4f}")
            
            print(f"   📉 Validation Loss:")
            print(f"      • Best:  {val_losses.min():.4f}")
            print(f"      • Mean:  {val_losses.mean():.4f}")
            print(f"      • Std:   {val_losses.std():.4f}")
            print(f"      • Max:   {val_losses.max():.4f}")
            
            # Best configurations
            best_acc_result = successful_results.loc[successful_results['best_val_acc'].idxmax()]
            best_loss_result = successful_results.loc[successful_results['best_val_loss'].idxmin()]
            
            print(f"\n🏆 BEST CONFIGURATIONS:")
            print(f"   🥇 Highest Accuracy ({best_acc_result['best_val_acc']:.4f}):")
            for param in model_grid.keys():
                if param in best_acc_result:
                    print(f"      {param}: {best_acc_result[param]}")
            
            print(f"   🥈 Lowest Loss ({best_loss_result['best_val_loss']:.4f}):")
            for param in model_grid.keys():
                if param in best_loss_result:
                    print(f"      {param}: {best_loss_result[param]}")
            
            # Top 5 configurations
            top5_results = successful_results.nlargest(5, 'best_val_acc')
            print(f"\n📈 TOP 5 CONFIGURATIONS:")
            for i, (_, result) in enumerate(top5_results.iterrows(), 1):
                print(f"   #{i}. Acc: {result['best_val_acc']:.4f} | Loss: {result['best_val_loss']:.4f}")
                param_str = " | ".join([f"{k}:{v}" for k, v in result.items() 
                                      if k in model_grid.keys()])
                print(f"      {param_str}")
        
        print(f"\n📁 RESULTS SAVED:")
        print(f"   📄 CSV: {os.path.join(args.output, 'grid_search_results.csv')}")
        print(f"   📄 JSON: {os.path.join(args.output, 'grid_search_results.json')}")
        print(f"   📊 Status: {os.path.join(args.output, 'grid_search_status.json')}")
        print(f"   📈 Plots: {os.path.join(args.output, 'plots/')}")
        print(f"      • top_performers.png - Top 10 configurations")
        print(f"      • parameter_accuracy.png - Parameter vs accuracy plots")
        print(f"      • performance_distribution.png - Performance distributions")
        print(f"      • parameter_correlation.png - Parameter correlation heatmap")
        print(f"      • parameter_importance.png - Parameter importance ranking")
        print(f"="*60)
        
    except Exception as e:
        print(f"❌ Grid search failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 