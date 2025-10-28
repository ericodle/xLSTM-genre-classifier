#!/usr/bin/env python3
"""
Model Training Script for GenreDiscern

Usage:
    python src/training/train_model.py --data gtzan-data/mfccs_splits --model GRU --output output_dir --lr 0.001
    python src/training/train_model.py --data gtzan-data/mfccs_splits --model xLSTM --output output_dir --lr 0.0001 --epochs 10 --batch-size 8
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.config import Config
from core.utils import setup_logging
from training.train import ModelTrainer


def _run_automatic_evaluation(
    trainer, data_path: str, output_dir: str, model_type: str, logger: logging.Logger
):
    """Run automatic evaluation and generate plots."""
    try:
        from eval.evaluator import ModelEvaluator

        logger.info("Starting automatic evaluation...")

        # Load class names from the data
        # Handle pre-split directory format
        if os.path.isdir(data_path):
            # Pre-split format: load from train.json
            train_json = os.path.join(data_path, "train.json")
            if os.path.exists(train_json):
                with open(train_json, "r") as f:
                    data = json.load(f)
                    class_names = data.get("mapping", [])
                logger.info(
                    f"Loaded class names from pre-split directory: {len(class_names)} classes"
                )
            else:
                raise ValueError(f"train.json not found in directory: {data_path}")
        else:
            # Single-file format
            with open(data_path, "r") as f:
                data = json.load(f)
                class_names = data.get("mapping", [])
            logger.info(f"Loaded class names from single file: {len(class_names)} classes")

        logger.info(f"Class names: {class_names}")

        # Get the trained model
        model = trainer.model
        if model is None:
            raise ValueError("No trained model found")

        logger.info("Model loaded successfully")

        # Create evaluator
        evaluator = ModelEvaluator(model, logger=logger)

        # Get test data loader
        test_loader = trainer.test_loader
        if test_loader is None:
            raise ValueError("No test data loader found")

        logger.info("Test data loader found")

        # Run evaluation
        logger.info("Running model evaluation...")
        evaluation_results = evaluator.evaluate_model(test_loader, class_names)

        # Create evaluation output directory
        eval_output_dir = os.path.join(output_dir, "evaluation")
        os.makedirs(eval_output_dir, exist_ok=True)
        logger.info(f"Created evaluation directory: {eval_output_dir}")

        # Generate plots
        logger.info("Generating evaluation plots...")
        evaluator.generate_evaluation_plots(evaluation_results, eval_output_dir, class_names)

        # Save results
        logger.info("Saving evaluation results...")
        evaluator.save_evaluation_results(
            evaluation_results, os.path.join(eval_output_dir, "evaluation_results.json")
        )

        logger.info("Automatic evaluation completed successfully")

    except Exception as e:
        logger.warning(f"Failed to generate evaluation plots: {e}")


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
    if data_path.endswith(".json"):
        try:
            with open(data_path, "r") as f:
                data = json.load(f)
                mapping = data.get("mapping", [])

                # Check for FMA-specific genres
                fma_genres = [
                    "Blues",
                    "Classical",
                    "Country",
                    "Easy Listening",
                    "Electronic",
                    "Experimental",
                    "Folk",
                    "Hip-Hop",
                    "Instrumental",
                    "International",
                    "Jazz",
                    "Old-Time / Historic",
                    "Pop",
                    "Rock",
                    "Soul-RnB",
                    "Spoken",
                ]

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
        description="GenreDiscern Model Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python src/training/train_model.py --data gtzan-data/mfccs_splits --model GRU --output output_dir --lr 0.001
    python src/training/train_model.py --data gtzan-data/mfccs_splits --model xLSTM --output output_dir --lr 0.0001 --epochs 10 --batch-size 8
        """,
    )

    # Required arguments
    parser.add_argument("--data", required=True, help="Path to MFCC data file")
    parser.add_argument("--model", required=True, help="Type of model to train")
    parser.add_argument("--output", required=True, help="Output directory")

    # Optional arguments
    parser.add_argument("--lr", "--learning-rate", type=float, help="Initial learning rate")
    parser.add_argument("--epochs", type=int, help="Maximum number of epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--hidden-size", type=int, help="Hidden size (RNN/GRU/LSTM)")
    parser.add_argument("--num-layers", type=int, help="Number of layers (RNN/GRU/LSTM)")
    parser.add_argument("--dropout", type=float, help="Dropout probability (0-1)")
    parser.add_argument(
        "--init",
        choices=["xavier", "kaiming", "orthogonal", "rnn"],
        help="Optional weight initializer to apply after model creation",
    )
    parser.add_argument(
        "--improvement-threshold",
        type=float,
        help="Early stopping improvement threshold (default: 0.0001)",
    )
    parser.add_argument("--patience", type=int, help="Early stopping patience (epochs)")
    parser.add_argument("--no-early-stopping", action="store_true", help="Disable early stopping")
    parser.add_argument("--gradient-clip", type=float, help="Gradient clipping norm (default: 1.0)")
    parser.add_argument(
        "--use-pretrained", action="store_true", help="Use pretrained ImageNet weights (for VGG16)"
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        help="Label smoothing factor (0.0-1.0, typical: 0.1) to reduce overfitting",
    )
    parser.add_argument(
        "--regression",
        action="store_true",
        help="Use regression mode (output membership scores directly, for CNN)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    return parser


def main():
    """Main entry point."""
    parser = setup_cli_parser()
    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level)

    logger.info("Starting model training...")
    logger.info(f"Data: {args.data}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Output: {args.output}")

    # Auto-detect dataset type and optimize config
    dataset_type = detect_dataset_type(args.data)
    config = Config()
    if dataset_type:
        logger.info(f"Detected dataset type: {dataset_type}")
        config.optimize_for_dataset(dataset_type, args.model)
        logger.info(f"Loaded optimized defaults for {dataset_type} dataset")

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
        kwargs["improvement_window"] = args.patience  # Map patience to improvement_window
        config.training.improvement_window = args.patience
        user_overrides.append(f"patience (improvement_window): {args.patience}")
    if args.no_early_stopping:
        kwargs["early_stopping"] = False
        config.training.early_stopping = False
        user_overrides.append("early_stopping: False")
    if args.gradient_clip:
        kwargs["gradient_clip_norm"] = args.gradient_clip
        config.training.gradient_clip_norm = args.gradient_clip
        user_overrides.append(f"gradient_clip_norm: {args.gradient_clip}")
    if args.use_pretrained:
        kwargs["pretrained"] = True
        config.model.pretrained = True
        user_overrides.append("pretrained: True")
    if args.label_smoothing is not None:
        kwargs["label_smoothing"] = args.label_smoothing
        config.model.label_smoothing = args.label_smoothing
        user_overrides.append(f"label_smoothing: {args.label_smoothing}")
    if args.regression:
        kwargs["regression_mode"] = True
        config.model.regression_mode = True
        user_overrides.append("regression_mode: True")

    # Log user overrides
    if user_overrides:
        logger.info(f"User parameter overrides applied: {', '.join(user_overrides)}")
        logger.info("Final training parameters will use user-specified values where provided")
    else:
        logger.info("Using optimized defaults for all parameters")

    # Train model
    try:
        # Create trainer with config
        trainer = ModelTrainer(config, logger)

        # Update config with user-provided parameters
        for key, value in kwargs.items():
            if hasattr(config.model, key):
                setattr(config.model, key, value)
                logger.debug(f"Updated model.{key} = {value}")
            elif hasattr(config.training, key):
                setattr(config.training, key, value)
                logger.debug(f"Updated training.{key} = {value}")

        # Setup training
        setup_kwargs = {}
        if "max_samples" in kwargs:
            setup_kwargs["max_samples"] = kwargs["max_samples"]
        if "memory_efficient" in kwargs:
            setup_kwargs["memory_efficient"] = kwargs["memory_efficient"]

        trainer.setup_training(
            data_path=args.data, model_type=args.model, output_dir=args.output, **setup_kwargs
        )

        # Train model
        training_history = trainer.train()

        # Evaluate model
        test_loss, test_acc = trainer._evaluate_model()

        # Run automatic evaluation to generate plots
        try:
            _run_automatic_evaluation(trainer, args.data, args.output, args.model, logger)
        except Exception as e:
            logger.warning(f"Failed to generate evaluation plots: {e}")

        logger.info("Training completed successfully")
        logger.info(f"Final test accuracy: {test_acc:.4f}")
        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
