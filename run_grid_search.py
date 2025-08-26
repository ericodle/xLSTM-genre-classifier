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
    if params_file and os.path.exists(params_file):
        with open(params_file, 'r') as f:
            return json.load(f)
    
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
    # Run GRU grid search with default parameters
    python run_grid_search.py --model GRU --data ./mfccs/gtzan_mfcc.json --output ./output/gru_gridsearch
    
    # Run LSTM grid search with custom parameters
    python run_grid_search.py --model LSTM --data ./mfccs/gtzan_mfcc.json --output ./output/lstm_gridsearch --params lstm_params.json
    
    # Run CNN grid search with custom output
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
    
    # Confirm before proceeding
    print(f"This will run {total_combinations} training sessions.")
    response = input("Continue? (y/N): ")
    if response.lower() != 'y':
        print("Grid search cancelled.")
        return
    
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
            param_grid=model_grid
        )
        
        # Display results summary
        successful_results = results[results['status'] == 'completed']
        failed_results = results[results['status'] == 'failed']
        
        print(f"\n✅ Grid search completed!")
        print(f"📊 Successful runs: {len(successful_results)}")
        print(f"❌ Failed runs: {len(failed_results)}")
        
        if not successful_results.empty:
            best_result = successful_results.loc[successful_results['best_val_acc'].idxmax()]
            print(f"\n🏆 Best configuration:")
            for param in model_grid.keys():
                if param in best_result:
                    print(f"   {param}: {best_result[param]}")
            print(f"   Best validation accuracy: {best_result['best_val_acc']:.4f}")
        
        print(f"\n📁 Results saved to: {args.output}")
        print(f"📄 CSV: {os.path.join(args.output, 'grid_search_results.csv')}")
        print(f"📄 JSON: {os.path.join(args.output, 'grid_search_results.json')}")
        
    except Exception as e:
        print(f"❌ Grid search failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 