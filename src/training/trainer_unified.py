"""
Unified training interface for GenreDiscern.

This module provides a single, consistent training interface that can be used
for both single model training and hyperparameter optimization (OFAT/Grid Search).
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.config import Config
from core.utils import setup_logging, ensure_directory
from training.trainer import ModelTrainer


class UnifiedTrainer:
    """
    Unified training interface that provides consistent training across all use cases.
    
    This class wraps the ModelTrainer to provide a single interface for:
    - Single model training
    - Grid search hyperparameter optimization
    - OFAT analysis
    - Any other training scenarios
    """
    
    def __init__(self, config: Optional[Config] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize unified trainer.
        
        Args:
            config: Configuration object (optional)
            logger: Logger instance (optional)
        """
        self.config = config or Config()
        self.logger = logger or setup_logging()
        self.trainer: Optional[ModelTrainer] = None
        
    def train_single_model(
        self,
        data_path: str,
        model_type: str,
        output_dir: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train a single model with specified parameters.
        
        This replaces the legacy train_model.py functionality with a unified approach.
        
        Args:
            data_path: Path to training data
            model_type: Type of model to train
            output_dir: Output directory for results
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training results and history
        """
        self.logger.info(f"Starting unified training for {model_type} model")
        self.logger.info(f"Data: {data_path}")
        self.logger.info(f"Output: {output_dir}")
        
        # Create trainer instance
        self.trainer = ModelTrainer(self.config, self.logger)
        
        # Setup training
        self.trainer.setup_training(
            data_path=data_path,
            model_type=model_type,
            output_dir=output_dir,
            **kwargs
        )
        
        # Train model
        training_history = self.trainer.train()
        
        # Get final evaluation results
        test_loss, test_acc = self.trainer._evaluate_model()
        
        # Run automatic evaluation to generate confusion matrix and KS curves
        try:
            from src.main import run_automatic_evaluation
            evaluation_results = run_automatic_evaluation(
                trainer=self.trainer,
                data_path=data_path,
                output_dir=output_dir,
                model_type=model_type,
                logger=self.logger
            )
            self.logger.info("Evaluation results generated successfully")
        except Exception as e:
            self.logger.warning(f"Failed to generate evaluation results: {e}")
            evaluation_results = None
        
        # Prepare results
        results = {
            "training_history": training_history,
            "final_test_accuracy": test_acc,
            "final_test_loss": test_loss,
            "model_type": model_type,
            "output_dir": output_dir,
            "config": self.config.get_model_config(),
            "evaluation_results": evaluation_results
        }
        
        self.logger.info(f"Training completed successfully")
        self.logger.info(f"Final test accuracy: {test_acc:.4f}")
        
        return results
    
    def train_with_parameters(
        self,
        data_path: str,
        model_type: str,
        output_dir: str,
        parameters: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train a model with specific parameter overrides.
        
        This is used by grid search and OFAT analysis.
        
        Args:
            data_path: Path to training data
            model_type: Type of model to train
            output_dir: Output directory for results
            parameters: Parameter overrides
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training results
        """
        # Create a copy of config to avoid modifying the original
        import copy
        config_override = copy.deepcopy(self.config)
        
        # Apply parameter overrides
        for param_name, param_value in parameters.items():
            if hasattr(config_override.model, param_name):
                setattr(config_override.model, param_name, param_value)
                self.logger.debug(f"Set model.{param_name} = {param_value}")
            elif hasattr(config_override.training, param_name):
                setattr(config_override.training, param_name, param_value)
                self.logger.debug(f"Set training.{param_name} = {param_value}")
            else:
                self.logger.warning(f"Unknown parameter: {param_name}")
        
        # Create trainer with overridden config
        trainer = ModelTrainer(config_override, self.logger)
        
        # Setup training
        trainer.setup_training(
            data_path=data_path,
            model_type=model_type,
            output_dir=output_dir,
            **kwargs
        )
        
        # Train model
        training_history = trainer.train()
        
        # Get final evaluation results
        test_loss, test_acc = trainer._evaluate_model()
        
        # Run automatic evaluation to generate confusion matrix and KS curves
        try:
            from src.main import run_automatic_evaluation
            evaluation_results = run_automatic_evaluation(
                trainer=trainer,
                data_path=data_path,
                output_dir=output_dir,
                model_type=model_type,
                logger=self.logger
            )
            self.logger.info("Evaluation results generated successfully")
        except Exception as e:
            self.logger.warning(f"Failed to generate evaluation results: {e}")
            evaluation_results = None
        
        # Prepare results
        results = {
            "training_history": training_history,
            "final_test_accuracy": test_acc,
            "final_test_loss": test_loss,
            "model_type": model_type,
            "output_dir": output_dir,
            "parameters": parameters,
            "config": config_override.get_model_config(),
            "evaluation_results": evaluation_results
        }
        
        return results
    
    def evaluate_model(
        self,
        model_path: str,
        data_path: str,
        output_dir: str,
        class_names: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model.
        
        Args:
            model_path: Path to trained model
            data_path: Path to evaluation data
            output_dir: Output directory for results
            class_names: List of class names for reporting
            
        Returns:
            Dictionary containing evaluation results
        """
        from training.evaluator import ModelEvaluator
        
        # Load model (this would need to be implemented based on your model loading needs)
        # For now, this is a placeholder
        self.logger.info(f"Evaluating model: {model_path}")
        
        # Create evaluator
        evaluator = ModelEvaluator(None, logger=self.logger)  # Model loading would go here
        
        # Run evaluation
        results = evaluator.evaluate_model(None, class_names)  # DataLoader would go here
        
        return results


def train_model_unified(
    data_path: str,
    model_type: str,
    output_dir: str,
    config_path: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Unified training function that replaces the legacy train_model.py main function.
    
    This provides a drop-in replacement for the old training script.
    
    Args:
        data_path: Path to training data
        model_type: Type of model to train
        output_dir: Output directory for results
        config_path: Path to configuration file
        **kwargs: Additional training parameters
        
    Returns:
        Dictionary containing training results
    """
    # Load configuration
    config = Config(config_path) if config_path else Config()
    
    # Create unified trainer
    trainer = UnifiedTrainer(config)
    
    # Train model
    results = trainer.train_single_model(
        data_path=data_path,
        model_type=model_type,
        output_dir=output_dir,
        **kwargs
    )
    
    return results


# Backward compatibility function
def main(mfcc_path: str, model_type: str, output_directory: str, initial_lr: float):
    """
    Backward compatibility function that matches the old train_model.py interface.
    
    This allows existing scripts to use the unified trainer without modification.
    """
    # Create config with learning rate override
    config = Config()
    config.model.learning_rate = initial_lr
    
    # Create unified trainer
    trainer = UnifiedTrainer(config)
    
    # Train model
    results = trainer.train_single_model(
        data_path=mfcc_path,
        model_type=model_type,
        output_dir=output_directory
    )
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified GenreDiscern Training")
    parser.add_argument("data_path", help="Path to training data")
    parser.add_argument("model_type", help="Type of model to train")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    
    args = parser.parse_args()
    
    # Create config
    config = Config(args.config) if args.config else Config()
    if args.learning_rate:
        config.model.learning_rate = args.learning_rate
    
    # Train model
    trainer = UnifiedTrainer(config)
    results = trainer.train_single_model(
        data_path=args.data_path,
        model_type=args.model_type,
        output_dir=args.output_dir
    )
    
    print(f"Training completed. Final accuracy: {results['final_test_accuracy']:.4f}")
