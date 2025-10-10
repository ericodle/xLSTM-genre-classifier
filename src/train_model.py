#!/usr/bin/env python3
"""
Unified Model Training Script for GenreDiscern

This is the single, unified training script that:
- Uses the same training framework as OFAT/Grid Search
- Eliminates code duplication
- Provides consistent behavior across all training scenarios
- Maintains backward compatibility
- Supports both legacy and modern CLI interfaces

Usage:
    # Legacy style (backward compatible)
    python src/train_model.py mfccs/gtzan_13.json CNN output_dir 0.001
    
    # Modern style with named arguments
    python src/train_model.py --data mfccs/gtzan_13.json --model CNN --output output_dir --lr 0.001
    
    # With configuration file
    python src/train_model.py --data mfccs/gtzan_13.json --model CNN --output output_dir --config config.json
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.trainer_unified import train_model_unified, main as unified_main
import json
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
    # Legacy style (backward compatible)
    python src/train_model.py mfccs/gtzan_13.json CNN output_dir 0.001
    
    # Modern style with named arguments
    python src/train_model.py --data mfccs/gtzan_13.json --model CNN --output output_dir --lr 0.001
    
    # With configuration file
    python src/train_model.py --data mfccs/gtzan_13.json --model CNN --output output_dir --config config.json
        """
    )
    
    # Positional arguments (for backward compatibility)
    parser.add_argument("mfcc_path", nargs="?", help="Path to MFCC data file")
    parser.add_argument("model_type", nargs="?", help="Type of model to train")
    parser.add_argument("output_directory", nargs="?", help="Output directory")
    parser.add_argument("initial_lr", nargs="?", type=float, help="Initial learning rate")
    
    # Named arguments
    parser.add_argument("--data", help="Path to MFCC data file")
    parser.add_argument("--model", help="Type of model to train")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--lr", "--learning-rate", type=float, help="Initial learning rate")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--epochs", type=int, help="Maximum number of epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--hidden-size", type=int, help="Hidden size (RNN/GRU/LSTM)")
    parser.add_argument("--num-layers", type=int, help="Number of layers (RNN/GRU/LSTM)")
    parser.add_argument("--dropout", type=float, help="Dropout probability (0-1)")
    parser.add_argument("--improvement-threshold", type=float, help="Early stopping improvement threshold (default: 0.0001)")
    parser.add_argument("--patience", type=int, help="Early stopping patience (epochs)")
    parser.add_argument("--no-early-stopping", action="store_true", help="Disable early stopping")
    parser.add_argument("--gradient-clip", type=float, help="Gradient clipping norm (default: 1.0)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    return parser


def main():
    """Main entry point with backward compatibility."""
    parser = setup_cli_parser()
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level)
    
    logger.info("Starting unified model training...")
    
    # Determine arguments (backward compatibility)
    if args.mfcc_path and args.model_type and args.output_directory and args.initial_lr:
        # Legacy positional arguments
        data_path = args.mfcc_path
        model_type = args.model_type
        output_dir = args.output_directory
        initial_lr = args.initial_lr
        
        logger.info("Using legacy argument style")
        logger.info(f"Data: {data_path}")
        logger.info(f"Model: {model_type}")
        logger.info(f"Output: {output_dir}")
        logger.info(f"Learning Rate: {initial_lr}")
        
        # Use legacy main function for backward compatibility
        try:
            results = unified_main(data_path, model_type, output_dir, initial_lr)
            logger.info("Training completed successfully")
            return 0
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return 1
            
    elif args.data and args.model and args.output:
        # New named arguments
        data_path = args.data
        model_type = args.model
        output_dir = args.output
        
        logger.info("Using named argument style")
        logger.info(f"Data: {data_path}")
        logger.info(f"Model: {model_type}")
        logger.info(f"Output: {output_dir}")
        
        # Auto-detect dataset type and optimize config
        dataset_type = detect_dataset_type(data_path)
        if dataset_type:
            logger.info(f"Detected dataset type: {dataset_type}")
            # Create config and optimize it
            config = Config(args.config) if args.config else Config()
            config.optimize_for_dataset(dataset_type, model_type)
            logger.info(f"Optimized config for {dataset_type} dataset")
        else:
            config = Config(args.config) if args.config else Config()
        
        # Prepare additional parameters
        kwargs = {}
        if args.lr:
            kwargs["learning_rate"] = args.lr
        if args.epochs:
            kwargs["max_epochs"] = args.epochs
        if args.batch_size:
            kwargs["batch_size"] = args.batch_size
        if args.hidden_size:
            kwargs["hidden_size"] = args.hidden_size
        if args.num_layers:
            kwargs["num_layers"] = args.num_layers
        if args.dropout is not None:
            kwargs["dropout"] = args.dropout
        if args.improvement_threshold:
            kwargs["improvement_threshold"] = args.improvement_threshold
        if args.patience:
            kwargs["patience"] = args.patience
        if args.no_early_stopping:
            kwargs["early_stopping"] = False
        if args.gradient_clip:
            kwargs["gradient_clip_norm"] = args.gradient_clip
        
        # Use unified training function
        try:
            # Create unified trainer with optimized config
            from training.trainer_unified import UnifiedTrainer
            trainer = UnifiedTrainer(config, logger)
            
            # Train model
            results = trainer.train_single_model(
                data_path=data_path,
                model_type=model_type,
                output_dir=output_dir,
                **kwargs
            )
            
            logger.info("Training completed successfully")
            logger.info(f"Final test accuracy: {results['final_test_accuracy']:.4f}")
            return 0
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return 1
    else:
        # No valid arguments provided
        parser.print_help()
        logger.error("No valid arguments provided")
        return 1


if __name__ == "__main__":
    sys.exit(main())
