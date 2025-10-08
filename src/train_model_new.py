#!/usr/bin/env python3
"""
Unified Model Training Script for GenreDiscern

This script replaces the legacy train_model.py with a unified approach that:
- Uses the same training framework as OFAT/Grid Search
- Eliminates code duplication
- Provides consistent behavior across all training scenarios
- Maintains backward compatibility

Usage:
    python src/train_model_unified.py mfccs/gtzan_13.json CNN output_dir 0.001
    python src/train_model_unified.py --data mfccs/gtzan_13.json --model CNN --output output_dir --lr 0.001
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.trainer_unified import train_model_unified, main as unified_main
from core.config import Config
from core.utils import setup_logging


def setup_cli_parser():
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Unified GenreDiscern Model Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Legacy style (backward compatible)
    python train_model_unified.py mfccs/gtzan_13.json CNN output_dir 0.001
    
    # New style with named arguments
    python train_model_unified.py --data mfccs/gtzan_13.json --model CNN --output output_dir --lr 0.001
    
    # With configuration file
    python train_model_unified.py --data mfccs/gtzan_13.json --model CNN --output output_dir --config config.json
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
        
        # Prepare additional parameters
        kwargs = {}
        if args.lr:
            kwargs["learning_rate"] = args.lr
        if args.epochs:
            kwargs["max_epochs"] = args.epochs
        if args.batch_size:
            kwargs["batch_size"] = args.batch_size
        
        # Use unified training function
        try:
            results = train_model_unified(
                data_path=data_path,
                model_type=model_type,
                output_dir=output_dir,
                config_path=args.config,
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
