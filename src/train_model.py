#!/usr/bin/env python3
"""
Unified Model Training Script for GenreDiscern

This is the single, unified training script that:
- Uses the same training framework as OFAT/Grid Search
- Eliminates code duplication
- Provides consistent behavior across all training scenarios
- Supports modern CLI interface only

Usage:
    python src/train_model.py --data mfccs/gtzan_13.json --model CNN --output output_dir --lr 0.001
    python src/train_model.py --data mfccs/gtzan_13.json --model xLSTM --output output_dir --lr 0.0001 --epochs 10 --batch-size 8
    python src/train_model.py --data mfccs/gtzan_13.json --model CNN --output output_dir --config config.json
"""

import sys
import os
import argparse
import logging
import json
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.trainer_unified import UnifiedTrainer
from core.config import Config
from core.utils import setup_logging


def detect_dataset_type(data_path: str) -> str:
    """Detect dataset type from data path or content."""
    if not data_path:
        return None
    
    # Check filename patterns
    if "fma" in data_path.lower():
        return "FMA"
    elif "gtzan" in data_path.lower():
        return "GTZAN"
    
    # Check file content if it's a JSON file
    if data_path.endswith('.json'):
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)
                mapping = data.get('mapping', [])
                
                # Check for FMA-specific genres
                fma_genres = ['Blues', 'Classical', 'Country', 'Easy Listening', 'Electronic', 
                             'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 
                             'Jazz', 'Old-Time / Historic', 'Pop', 'Rock', 'Soul-RnB', 'Spoken']
                
                if any(genre in mapping for genre in fma_genres):
                    return "FMA"
                elif len(mapping) == 10:  # GTZAN has 10 genres
                    return "GTZAN"
        except Exception:
            pass
    
    return None


def setup_cli_parser():
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Unified GenreDiscern Model Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python src/train_model.py --data mfccs/gtzan_13.json --model CNN --output output_dir --lr 0.001
    python src/train_model.py --data mfccs/gtzan_13.json --model xLSTM --output output_dir --lr 0.0001 --epochs 10 --batch-size 8
    python src/train_model.py --data mfccs/gtzan_13.json --model CNN --output output_dir --config config.json
        """
    )
    
    # Required arguments
    parser.add_argument("--data", required=True, help="Path to MFCC data file")
    parser.add_argument("--model", required=True, help="Type of model to train")
    parser.add_argument("--output", required=True, help="Output directory")
    
    # Optional arguments
    parser.add_argument("--lr", "--learning-rate", type=float, help="Initial learning rate")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--epochs", type=int, help="Maximum number of epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--hidden-size", type=int, help="Hidden size (RNN/GRU/LSTM)")
    parser.add_argument("--num-layers", type=int, help="Number of layers (RNN/GRU/LSTM)")
    parser.add_argument("--dropout", type=float, help="Dropout probability (0-1)")
    parser.add_argument("--init", choices=["xavier", "kaiming", "orthogonal", "rnn"], help="Optional weight initializer to apply after model creation")
    parser.add_argument("--improvement-threshold", type=float, help="Early stopping improvement threshold (default: 0.0001)")
    parser.add_argument("--patience", type=int, help="Early stopping patience (epochs)")
    parser.add_argument("--no-early-stopping", action="store_true", help="Disable early stopping")
    parser.add_argument("--gradient-clip", type=float, help="Gradient clipping norm (default: 1.0)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    return parser


def main():
    """Main entry point."""
    parser = setup_cli_parser()
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level)
    
    logger.info("Starting unified model training...")
    logger.info(f"Data: {args.data}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Output: {args.output}")
    
    # Auto-detect dataset type and optimize config
    dataset_type = detect_dataset_type(args.data)
    if dataset_type:
        logger.info(f"Detected dataset type: {dataset_type}")
        # Create config and optimize it
        config = Config(args.config) if args.config else Config()
        config.optimize_for_dataset(dataset_type, args.model)
        logger.info(f"Optimized config for {dataset_type} dataset")
    else:
        config = Config(args.config) if args.config else Config()
    
    # Prepare additional parameters - these will override optimized defaults
    kwargs = {}
    user_overrides = []
    
    if args.lr:
        kwargs["learning_rate"] = args.lr
        config.model.learning_rate = args.lr
        user_overrides.append(f"learning_rate: {args.lr}")
    if args.epochs:
        kwargs["max_epochs"] = args.epochs
        config.model.max_epochs = args.epochs
        user_overrides.append(f"max_epochs: {args.epochs}")
    if args.batch_size:
        kwargs["batch_size"] = args.batch_size
        config.model.batch_size = args.batch_size
        user_overrides.append(f"batch_size: {args.batch_size}")
    if args.hidden_size:
        kwargs["hidden_size"] = args.hidden_size
        config.model.hidden_size = args.hidden_size
        user_overrides.append(f"hidden_size: {args.hidden_size}")
    if args.num_layers:
        kwargs["num_layers"] = args.num_layers
        config.model.num_layers = args.num_layers
        user_overrides.append(f"num_layers: {args.num_layers}")
    if args.dropout is not None:
        kwargs["dropout"] = args.dropout
        config.model.dropout = args.dropout
        user_overrides.append(f"dropout: {args.dropout}")
    if args.init:
        kwargs["init"] = args.init
        config.model.init = args.init
        user_overrides.append(f"init: {args.init}")
    if args.improvement_threshold:
        kwargs["improvement_threshold"] = args.improvement_threshold
        config.training.improvement_threshold = args.improvement_threshold
        user_overrides.append(f"improvement_threshold: {args.improvement_threshold}")
    if args.patience:
        kwargs["patience"] = args.patience
        config.model.early_stopping_patience = args.patience
        user_overrides.append(f"patience: {args.patience}")
    if args.no_early_stopping:
        kwargs["early_stopping"] = False
        config.training.early_stopping = False
        user_overrides.append("early_stopping: False")
    if args.gradient_clip:
        kwargs["gradient_clip_norm"] = args.gradient_clip
        config.training.gradient_clip_norm = args.gradient_clip
        user_overrides.append(f"gradient_clip_norm: {args.gradient_clip}")
    
    # Log user overrides
    if user_overrides:
        logger.info(f"User parameter overrides: {', '.join(user_overrides)}")
    else:
        logger.info("Using optimized defaults for all parameters")
    
    # Use unified training function
    try:
        # Create unified trainer with optimized config
        trainer = UnifiedTrainer(config, logger)
        
        # Train model
        results = trainer.train_single_model(
            data_path=args.data,
            model_type=args.model,
            output_dir=args.output,
            **kwargs
        )
        
        logger.info("Training completed successfully")
        logger.info(f"Final test accuracy: {results['final_test_accuracy']:.4f}")
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())