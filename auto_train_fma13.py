#!/usr/bin/env python3
"""
Automated Training Script for FMA-13 Dataset

This script automatically trains all supported models on the FMA-13 dataset:
- FC (Fully Connected)
- CNN (Convolutional Neural Network)
- GRU (Gated Recurrent Unit)
- LSTM (Long Short-Term Memory)
- xLSTM (Extended LSTM)
- Transformer
- SVM (Support Vector Machine)

Each model is trained with optimized hyperparameters for the FMA-13 dataset.
"""

import os
import sys
import subprocess
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import argparse

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('auto_train_fma13.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutoTrainer:
    """Automated trainer for all models on FMA-13 dataset."""
    
    def __init__(self, data_path: str, base_output_dir: str, dry_run: bool = False):
        self.data_path = data_path
        self.base_output_dir = Path(base_output_dir)
        self.dry_run = dry_run
        self.results = {}
        self.start_time = datetime.now()
        
        # Create base output directory
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model configurations optimized for FMA-13
        self.model_configs = {
            "FC": {
                "lr": 0.001,
                "batch_size": 64,
                "epochs": 100,
                "dropout": 0.3,
                "hidden_dims": [512, 256, 128]
            },
            "CNN": {
                "lr": 0.0005,
                "batch_size": 32,
                "epochs": 150,
                "dropout": 0.3,
                "conv_layers": 4,
                "base_filters": 32,
                "kernel_size": 3,
                "pool_size": 2,
                "fc_hidden": 128
            },
            "GRU": {
                "lr": 0.001,
                "batch_size": 32,
                "epochs": 100,
                "dropout": 0.2,
                "hidden_size": 128,
                "num_layers": 2
            },
            "LSTM": {
                "lr": 0.001,
                "batch_size": 32,
                "epochs": 100,
                "dropout": 0.2,
                "hidden_size": 128,
                "num_layers": 2
            },
            "xLSTM": {
                "lr": 0.0008,
                "batch_size": 32,
                "epochs": 120,
                "dropout": 0.15,
                "hidden_size": 128,
                "num_layers": 2
            },
            "Transformer": {
                "lr": 0.0003,
                "batch_size": 16,
                "epochs": 100,
                "dropout": 0.1,
                "hidden_size": 128,
                "num_layers": 3,
                "num_heads": 8,
                "ff_dim": 256
            },
            "VGG16": {
                "lr": 0.0001,
                "batch_size": 16,
                "epochs": 200,
                "dropout": 0.5
            }
        }
        
        # SVM configurations
        self.svm_configs = {
            "SVM_RBF": {
                "kernel": "rbf",
                "C": 10.0,
                "gamma": "scale",
                "class_weight": "balanced"
            },
            "SVM_RBF_PCA": {
                "kernel": "rbf",
                "C": 10.0,
                "gamma": "scale",
                "class_weight": "balanced",
                "pca": 100
            },
            "SVM_Linear": {
                "kernel": "linear",
                "C": 1.0,
                "class_weight": "balanced"
            }
        }
    
    def run_training(self, model_type: str, config: Dict, output_dir: Path) -> Tuple[bool, str]:
        """Run training for a single model."""
        logger.info(f"Starting training for {model_type}...")
        
        if self.dry_run:
            logger.info(f"DRY RUN: Would train {model_type} with config: {config}")
            return True, "Dry run completed"
        
        try:
            # Build command for neural network models
            if model_type in ["FC", "CNN", "GRU", "LSTM", "xLSTM", "Transformer", "VGG16"]:
                cmd = [
                    "python", "src/train_model.py",
                    "--data", self.data_path,
                    "--model", model_type,
                    "--output", str(output_dir),
                    "--lr", str(config["lr"]),
                    "--batch-size", str(config["batch_size"]),
                    "--epochs", str(config["epochs"]),
                    "--dropout", str(config["dropout"])
                ]
                
                # Add model-specific parameters
                if model_type in ["GRU", "LSTM", "xLSTM", "Transformer"]:
                    cmd.extend(["--hidden-size", str(config["hidden_size"])])
                    cmd.extend(["--num-layers", str(config["num_layers"])])
                
                if model_type == "Transformer":
                    cmd.extend(["--num-heads", str(config["num_heads"])])
                    cmd.extend(["--ff-dim", str(config["ff_dim"])])
                
                if model_type == "CNN":
                    cmd.extend(["--conv-layers", str(config["conv_layers"])])
                    cmd.extend(["--base-filters", str(config["base_filters"])])
                    cmd.extend(["--kernel-size", str(config["kernel_size"])])
                    cmd.extend(["--pool-size", str(config["pool_size"])])
                    cmd.extend(["--fc-hidden", str(config["fc_hidden"])])
            
            # Build command for SVM models
            elif model_type.startswith("SVM"):
                cmd = [
                    "python", "src/training/train_svm.py",
                    "--data", self.data_path,
                    "--output", str(output_dir),
                    "--kernel", config["kernel"],
                    "--C", str(config["C"]),
                    "--gamma", str(config["gamma"]) if "gamma" in config else "scale",
                    "--class-weight", config["class_weight"]
                ]
                
                if "pca" in config and config["pca"] > 0:
                    cmd.extend(["--pca", str(config["pca"])])
            
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
            # Run the training command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )
            
            if result.returncode == 0:
                logger.info(f"✅ {model_type} training completed successfully")
                return True, "Training completed successfully"
            else:
                error_msg = f"Training failed with return code {result.returncode}\nSTDERR: {result.stderr}"
                logger.error(f"❌ {model_type} training failed: {error_msg}")
                return False, error_msg
                
        except subprocess.TimeoutExpired:
            error_msg = f"Training timed out after 2 hours"
            logger.error(f"⏰ {model_type} {error_msg}")
            return False, error_msg
        except Exception as e:
            error_msg = f"Training failed with exception: {str(e)}"
            logger.error(f"💥 {model_type} {error_msg}")
            return False, error_msg
    
    def train_all_models(self) -> Dict[str, Dict]:
        """Train all models and return results."""
        logger.info("🚀 Starting automated training for all models on FMA-13 dataset")
        logger.info(f"Data path: {self.data_path}")
        logger.info(f"Output directory: {self.base_output_dir}")
        logger.info(f"Dry run: {self.dry_run}")
        
        # Train neural network models
        for model_type, config in self.model_configs.items():
            output_dir = self.base_output_dir / f"{model_type.lower()}-fma-run"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            start_time = time.time()
            success, message = self.run_training(model_type, config, output_dir)
            end_time = time.time()
            
            self.results[model_type] = {
                "success": success,
                "message": message,
                "duration": end_time - start_time,
                "output_dir": str(output_dir),
                "config": config
            }
            
            if success:
                logger.info(f"✅ {model_type} completed in {end_time - start_time:.1f} seconds")
            else:
                logger.error(f"❌ {model_type} failed: {message}")
        
        # Train SVM models
        for model_type, config in self.svm_configs.items():
            output_dir = self.base_output_dir / f"{model_type.lower()}-fma"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            start_time = time.time()
            success, message = self.run_training(model_type, config, output_dir)
            end_time = time.time()
            
            self.results[model_type] = {
                "success": success,
                "message": message,
                "duration": end_time - start_time,
                "output_dir": str(output_dir),
                "config": config
            }
            
            if success:
                logger.info(f"✅ {model_type} completed in {end_time - start_time:.1f} seconds")
            else:
                logger.error(f"❌ {model_type} failed: {message}")
        
        return self.results
    
    def generate_summary_report(self) -> None:
        """Generate a summary report of all training results."""
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        successful_models = [name for name, result in self.results.items() if result["success"]]
        failed_models = [name for name, result in self.results.items() if not result["success"]]
        
        report = {
            "summary": {
                "total_models": len(self.results),
                "successful": len(successful_models),
                "failed": len(failed_models),
                "total_time_seconds": total_time,
                "total_time_hours": total_time / 3600,
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat()
            },
            "successful_models": successful_models,
            "failed_models": failed_models,
            "detailed_results": self.results
        }
        
        # Save detailed report
        report_path = self.base_output_dir / "training_summary.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary to console
        logger.info("\n" + "="*80)
        logger.info("🎯 TRAINING SUMMARY REPORT")
        logger.info("="*80)
        logger.info(f"Total models trained: {len(self.results)}")
        logger.info(f"Successful: {len(successful_models)}")
        logger.info(f"Failed: {len(failed_models)}")
        logger.info(f"Total time: {total_time/3600:.2f} hours")
        logger.info(f"Report saved to: {report_path}")
        
        if successful_models:
            logger.info(f"\n✅ Successful models: {', '.join(successful_models)}")
        
        if failed_models:
            logger.info(f"\n❌ Failed models: {', '.join(failed_models)}")
            for model in failed_models:
                logger.info(f"   - {model}: {self.results[model]['message']}")
        
        logger.info("="*80)
    
    def run_evaluation(self) -> None:
        """Run evaluation on all successfully trained models."""
        logger.info("\n🔍 Running evaluation on all trained models...")
        
        for model_name, result in self.results.items():
            if result["success"]:
                output_dir = Path(result["output_dir"])
                model_path = output_dir / "model.onnx"
                eval_dir = output_dir / "evaluation"
                
                if model_path.exists():
                    logger.info(f"Evaluating {model_name}...")
                    
                    if self.dry_run:
                        logger.info(f"DRY RUN: Would evaluate {model_name}")
                        continue
                    
                    try:
                        cmd = [
                            "python", "evaluate_model.py",
                            str(model_path),
                            self.data_path,
                            str(eval_dir)
                        ]
                        
                        eval_result = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            timeout=1800  # 30 minute timeout
                        )
                        
                        if eval_result.returncode == 0:
                            logger.info(f"✅ {model_name} evaluation completed")
                        else:
                            logger.error(f"❌ {model_name} evaluation failed: {eval_result.stderr}")
                    
                    except Exception as e:
                        logger.error(f"💥 {model_name} evaluation error: {str(e)}")
                else:
                    logger.warning(f"⚠️  Model file not found for {model_name}: {model_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Automated training script for all models on FMA-13 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all models (production)
    python auto_train_fma13.py --data mfccs/fma_13.json --output outputs/auto_fma13
    
    # Dry run to see what would be executed
    python auto_train_fma13.py --data mfccs/fma_13.json --output outputs/auto_fma13 --dry-run
    
    # Run only specific models
    python auto_train_fma13.py --data mfccs/fma_13.json --output outputs/auto_fma13 --models FC CNN GRU
        """
    )
    
    parser.add_argument(
        "--data", 
        required=True, 
        help="Path to FMA-13 MFCC data file (e.g., mfccs/fma_13.json)"
    )
    parser.add_argument(
        "--output", 
        required=True, 
        help="Base output directory for all model results"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Show what would be executed without actually running training"
    )
    parser.add_argument(
        "--models", 
        nargs="+", 
        help="Specific models to train (default: all models)",
        choices=["FC", "CNN", "GRU", "LSTM", "xLSTM", "Transformer", "VGG16", "SVM_RBF", "SVM_RBF_PCA", "SVM_Linear"]
    )
    parser.add_argument(
        "--skip-evaluation", 
        action="store_true", 
        help="Skip evaluation step after training"
    )
    
    args = parser.parse_args()
    
    # Validate data file exists
    if not os.path.exists(args.data):
        logger.error(f"Data file not found: {args.data}")
        sys.exit(1)
    
    # Create trainer
    trainer = AutoTrainer(args.data, args.output, args.dry_run)
    
    # Filter models if specified
    if args.models:
        filtered_configs = {}
        for model in args.models:
            if model in trainer.model_configs:
                filtered_configs[model] = trainer.model_configs[model]
            elif model in trainer.svm_configs:
                filtered_configs[model] = trainer.svm_configs[model]
            else:
                logger.warning(f"Unknown model: {model}")
        
        if filtered_configs:
            trainer.model_configs = {k: v for k, v in trainer.model_configs.items() if k in args.models}
            trainer.svm_configs = {k: v for k, v in trainer.svm_configs.items() if k in args.models}
        else:
            logger.error("No valid models specified")
            sys.exit(1)
    
    try:
        # Run training
        results = trainer.train_all_models()
        
        # Generate summary report
        trainer.generate_summary_report()
        
        # Run evaluation unless skipped
        if not args.skip_evaluation:
            trainer.run_evaluation()
        
        # Exit with appropriate code
        failed_count = sum(1 for r in results.values() if not r["success"])
        if failed_count == 0:
            logger.info("🎉 All models trained successfully!")
            sys.exit(0)
        else:
            logger.warning(f"⚠️  {failed_count} models failed. Check logs for details.")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("🛑 Training interrupted by user")
        trainer.generate_summary_report()
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Fatal error: {str(e)}")
        trainer.generate_summary_report()
        sys.exit(1)


if __name__ == "__main__":
    main()
