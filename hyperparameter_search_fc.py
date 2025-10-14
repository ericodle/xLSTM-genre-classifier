#!/usr/bin/env python3
"""
Hyperparameter Search for FC Model on GTZAN-13 Dataset

This script performs a comprehensive hyperparameter search for the FC (Fully Connected)
model on the GTZAN-13 dataset, testing different combinations of:
- Learning rates
- Batch sizes
- Hidden layer configurations
- Dropout rates
- Optimizers
- Weight decay values
"""

import os
import sys
import json
import logging
import itertools
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import pandas as pd

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from training.trainer_unified import UnifiedTrainer
from core.config import Config
from core.utils import setup_logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hyperparameter_search_fc.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FCHyperparameterSearch:
    """Hyperparameter search for FC model on GTZAN-13."""
    
    def __init__(self, data_path: str, base_output_dir: str):
        self.data_path = data_path
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize trainer
        self.config = Config()
        self.trainer = UnifiedTrainer(self.config, logger)
        
        # Define hyperparameter search space
        self.search_space = {
            # Learning rates
            'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01],
            
            # Batch sizes
            'batch_size': [16, 32, 64, 128],
            
            # Hidden layer configurations (as lists)
            'hidden_dims': [
                [256],           # Single layer
                [512],           # Single layer, larger
                [256, 128],      # Two layers
                [512, 256],      # Two layers, larger
                [512, 256, 128], # Three layers
                [1024, 512],     # Two layers, very large
                [512, 256, 128, 64], # Four layers
            ],
            
            # Dropout rates
            'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
            
            # Optimizers
            'optimizer': ['adam', 'adamw', 'sgd'],
            
            # Weight decay
            'weight_decay': [1e-5, 1e-4, 1e-3],
            
            # Learning rate scheduler
            'lr_scheduler': [True, False],
        }
        
        # Results storage
        self.results = []
        self.best_result = None
        self.start_time = datetime.now()
    
    def generate_parameter_combinations(self, max_combinations: int = 100) -> List[Dict[str, Any]]:
        """Generate parameter combinations for grid search."""
        logger.info("Generating parameter combinations...")
        
        # Get all parameter names and values
        param_names = list(self.search_space.keys())
        param_values = list(self.search_space.values())
        
        # Generate all combinations
        all_combinations = list(itertools.product(*param_values))
        
        # Convert to dictionaries
        combinations = []
        for combo in all_combinations:
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)
        
        # Limit number of combinations if too many
        if len(combinations) > max_combinations:
            logger.info(f"Too many combinations ({len(combinations)}), sampling {max_combinations} random combinations")
            import random
            random.seed(42)  # For reproducibility
            combinations = random.sample(combinations, max_combinations)
        
        logger.info(f"Generated {len(combinations)} parameter combinations")
        return combinations
    
    def run_single_experiment(self, params: Dict[str, Any], experiment_id: int) -> Dict[str, Any]:
        """Run a single hyperparameter experiment."""
        logger.info(f"Running experiment {experiment_id}: {params}")
        
        # Create output directory for this experiment
        exp_dir = self.base_output_dir / f"fc_gtzan_exp_{experiment_id:03d}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Train model with these parameters
            result = self.trainer.train_with_parameters(
                data_path=self.data_path,
                model_type="FC",
                output_dir=str(exp_dir),
                parameters=params
            )
            
            # Extract key metrics
            experiment_result = {
                'experiment_id': experiment_id,
                'parameters': params,
                'output_dir': str(exp_dir),
                'final_test_accuracy': result.get('final_test_accuracy', 0.0),
                'final_validation_accuracy': result.get('final_validation_accuracy', 0.0),
                'best_validation_accuracy': result.get('best_validation_accuracy', 0.0),
                'total_epochs': result.get('total_epochs', 0),
                'training_time': result.get('training_time', 0.0),
                'success': True,
                'error_message': None
            }
            
            logger.info(f"✅ Experiment {experiment_id} completed - Test Acc: {experiment_result['final_test_accuracy']:.4f}")
            return experiment_result
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Experiment {experiment_id} failed: {error_msg}")
            
            return {
                'experiment_id': experiment_id,
                'parameters': params,
                'output_dir': str(exp_dir),
                'final_test_accuracy': 0.0,
                'final_validation_accuracy': 0.0,
                'best_validation_accuracy': 0.0,
                'total_epochs': 0,
                'training_time': 0.0,
                'success': False,
                'error_message': error_msg
            }
    
    def run_hyperparameter_search(self, max_combinations: int = 50) -> None:
        """Run the complete hyperparameter search."""
        logger.info("🚀 Starting FC hyperparameter search on GTZAN-13")
        logger.info(f"Data path: {self.data_path}")
        logger.info(f"Output directory: {self.base_output_dir}")
        
        # Generate parameter combinations
        combinations = self.generate_parameter_combinations(max_combinations)
        
        # Run experiments
        for i, params in enumerate(combinations, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Experiment {i}/{len(combinations)}")
            logger.info(f"{'='*60}")
            
            result = self.run_single_experiment(params, i)
            self.results.append(result)
            
            # Update best result
            if result['success'] and (self.best_result is None or 
                                   result['final_test_accuracy'] > self.best_result['final_test_accuracy']):
                self.best_result = result
                logger.info(f"🏆 New best result! Test Accuracy: {result['final_test_accuracy']:.4f}")
        
        # Generate final report
        self.generate_final_report()
    
    def generate_final_report(self) -> None:
        """Generate comprehensive final report."""
        total_time = (datetime.now() - self.start_time).total_seconds()
        successful_experiments = [r for r in self.results if r['success']]
        failed_experiments = [r for r in self.results if not r['success']]
        
        # Create results DataFrame
        df = pd.DataFrame(self.results)
        
        # Save detailed results
        results_file = self.base_output_dir / "hyperparameter_search_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save CSV for easy analysis
        csv_file = self.base_output_dir / "hyperparameter_search_results.csv"
        df.to_csv(csv_file, index=False)
        
        # Generate summary report
        report = {
            "summary": {
                "total_experiments": len(self.results),
                "successful": len(successful_experiments),
                "failed": len(failed_experiments),
                "total_time_hours": total_time / 3600,
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat()
            },
            "best_result": self.best_result,
            "top_10_results": sorted(successful_experiments, 
                                   key=lambda x: x['final_test_accuracy'], 
                                   reverse=True)[:10]
        }
        
        # Save summary report
        summary_file = self.base_output_dir / "hyperparameter_search_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary to console
        logger.info("\n" + "="*80)
        logger.info("🎯 HYPERPARAMETER SEARCH SUMMARY")
        logger.info("="*80)
        logger.info(f"Total experiments: {len(self.results)}")
        logger.info(f"Successful: {len(successful_experiments)}")
        logger.info(f"Failed: {len(failed_experiments)}")
        logger.info(f"Total time: {total_time/3600:.2f} hours")
        
        if self.best_result:
            logger.info(f"\n🏆 BEST RESULT:")
            logger.info(f"   Test Accuracy: {self.best_result['final_test_accuracy']:.4f}")
            logger.info(f"   Parameters: {self.best_result['parameters']}")
            logger.info(f"   Output Dir: {self.best_result['output_dir']}")
        
        logger.info(f"\n📊 Results saved to:")
        logger.info(f"   Detailed: {results_file}")
        logger.info(f"   CSV: {csv_file}")
        logger.info(f"   Summary: {summary_file}")
        logger.info("="*80)
    
    def analyze_results(self) -> None:
        """Generate analysis plots and insights."""
        if not self.results:
            logger.warning("No results to analyze")
            return
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            df = pd.DataFrame(self.results)
            successful_df = df[df['success'] == True]
            
            if len(successful_df) == 0:
                logger.warning("No successful experiments to analyze")
                return
            
            # Create analysis plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('FC Model Hyperparameter Search Analysis', fontsize=16)
            
            # 1. Learning rate vs accuracy
            ax1 = axes[0, 0]
            for lr in successful_df['parameters'].apply(lambda x: x['learning_rate']).unique():
                lr_data = successful_df[successful_df['parameters'].apply(lambda x: x['learning_rate']) == lr]
                ax1.scatter(lr_data['parameters'].apply(lambda x: x['learning_rate']), 
                           lr_data['final_test_accuracy'], 
                           label=f'LR={lr}', alpha=0.7)
            ax1.set_xlabel('Learning Rate')
            ax1.set_ylabel('Test Accuracy')
            ax1.set_title('Learning Rate vs Test Accuracy')
            ax1.legend()
            ax1.set_xscale('log')
            
            # 2. Batch size vs accuracy
            ax2 = axes[0, 1]
            for bs in successful_df['parameters'].apply(lambda x: x['batch_size']).unique():
                bs_data = successful_df[successful_df['parameters'].apply(lambda x: x['batch_size']) == bs]
                ax2.scatter(bs_data['parameters'].apply(lambda x: x['batch_size']), 
                           bs_data['final_test_accuracy'], 
                           label=f'BS={bs}', alpha=0.7)
            ax2.set_xlabel('Batch Size')
            ax2.set_ylabel('Test Accuracy')
            ax2.set_title('Batch Size vs Test Accuracy')
            ax2.legend()
            
            # 3. Dropout vs accuracy
            ax3 = axes[1, 0]
            for dropout in successful_df['parameters'].apply(lambda x: x['dropout']).unique():
                drop_data = successful_df[successful_df['parameters'].apply(lambda x: x['dropout']) == dropout]
                ax3.scatter(drop_data['parameters'].apply(lambda x: x['dropout']), 
                           drop_data['final_test_accuracy'], 
                           label=f'Dropout={dropout}', alpha=0.7)
            ax3.set_xlabel('Dropout Rate')
            ax3.set_ylabel('Test Accuracy')
            ax3.set_title('Dropout vs Test Accuracy')
            ax3.legend()
            
            # 4. Accuracy distribution
            ax4 = axes[1, 1]
            ax4.hist(successful_df['final_test_accuracy'], bins=20, alpha=0.7, edgecolor='black')
            ax4.set_xlabel('Test Accuracy')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Distribution of Test Accuracies')
            
            plt.tight_layout()
            analysis_file = self.base_output_dir / "hyperparameter_analysis.png"
            plt.savefig(analysis_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"📈 Analysis plots saved to: {analysis_file}")
            
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available for analysis plots")
        except Exception as e:
            logger.error(f"Error generating analysis plots: {e}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for FC model on GTZAN-13",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full hyperparameter search
    python hyperparameter_search_fc.py --data mfccs/gtzan_13.json --output outputs/fc_hyperparameter_search
    
    # Run with limited combinations
    python hyperparameter_search_fc.py --data mfccs/gtzan_13.json --output outputs/fc_search --max-combinations 25
        """
    )
    
    parser.add_argument(
        "--data", 
        required=True, 
        help="Path to GTZAN-13 MFCC data file"
    )
    parser.add_argument(
        "--output", 
        required=True, 
        help="Output directory for hyperparameter search results"
    )
    parser.add_argument(
        "--max-combinations", 
        type=int, 
        default=50,
        help="Maximum number of parameter combinations to test (default: 50)"
    )
    
    args = parser.parse_args()
    
    # Validate data file exists
    if not os.path.exists(args.data):
        logger.error(f"Data file not found: {args.data}")
        sys.exit(1)
    
    # Create and run hyperparameter search
    search = FCHyperparameterSearch(args.data, args.output)
    
    try:
        search.run_hyperparameter_search(args.max_combinations)
        search.analyze_results()
        logger.info("🎉 Hyperparameter search completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("🛑 Hyperparameter search interrupted by user")
        search.generate_final_report()
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Fatal error: {str(e)}")
        search.generate_final_report()
        sys.exit(1)


if __name__ == "__main__":
    main()
