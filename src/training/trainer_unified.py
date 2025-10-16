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
        # Logging is handled by the calling script to avoid duplication
        
        # Create trainer instance
        self.trainer = ModelTrainer(self.config, self.logger)
        
        # Update config with any provided parameters
        for key, value in kwargs.items():
            if hasattr(self.config.model, key):
                setattr(self.config.model, key, value)
                self.logger.debug(f"Updated model.{key} = {value}")
            elif hasattr(self.config.training, key):
                setattr(self.config.training, key, value)
                self.logger.debug(f"Updated training.{key} = {value}")
            else:
                self.logger.debug(f"Parameter {key} not found in config, skipping")
        
        # Setup training (only pass supported parameters)
        setup_kwargs = {}
        if 'max_samples' in kwargs:
            setup_kwargs['max_samples'] = kwargs['max_samples']
        if 'memory_efficient' in kwargs:
            setup_kwargs['memory_efficient'] = kwargs['memory_efficient']
            
        self.trainer.setup_training(
            data_path=data_path,
            model_type=model_type,
            output_dir=output_dir,
            **setup_kwargs
        )
        
        # Train model
        training_history = self.trainer.train()
        
        # Get final evaluation results
        test_loss, test_acc = self.trainer._evaluate_model()
        
        # Run automatic evaluation to generate confusion matrix and KS curves
        try:
            evaluation_results = self._run_automatic_evaluation(
                data_path=data_path,
                output_dir=output_dir,
                model_type=model_type
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
        
        return results
    
    def _run_automatic_evaluation(
        self, 
        data_path: str, 
        output_dir: str, 
        model_type: str
    ) -> Dict[str, Any]:
        """Run automatic evaluation and generate plots."""
        try:
            from training.evaluator import ModelEvaluator
            import json
            
            self.logger.info("Starting automatic evaluation...")
            
            # Load the dataset to get class names
            with open(data_path, 'r') as f:
                data = json.load(f)
                class_names = data.get('mapping', [])
            
            self.logger.info(f"Found {len(class_names)} classes: {class_names}")
            
            # Get the trained model
            model = self.trainer.model
            if model is None:
                raise ValueError("No trained model found")
            
            self.logger.info("Model loaded successfully")
            
            # Create evaluator
            evaluator = ModelEvaluator(model, logger=self.logger)
            
            # Get test data loader
            test_loader = self.trainer.test_loader
            if test_loader is None:
                raise ValueError("No test data loader found")
            
            self.logger.info("Test data loader found")
            
            # Run evaluation
            self.logger.info("Running model evaluation...")
            evaluation_results = evaluator.evaluate_model(test_loader, class_names)
            
            # Create evaluation output directory
            eval_output_dir = os.path.join(output_dir, "evaluation")
            os.makedirs(eval_output_dir, exist_ok=True)
            self.logger.info(f"Created evaluation directory: {eval_output_dir}")
            
            # Generate plots
            self.logger.info("Generating evaluation plots...")
            evaluator.generate_evaluation_plots(
                evaluation_results, 
                eval_output_dir, 
                class_names
            )
            
            # Save results
            self.logger.info("Saving evaluation results...")
            evaluator.save_evaluation_results(
                evaluation_results, 
                os.path.join(eval_output_dir, "evaluation_results.json")
            )
            
            self.logger.info("Automatic evaluation completed successfully")
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Error in automatic evaluation: {str(e)}")
            self.logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
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
    logger: Optional[logging.Logger] = None,
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
        logger: Logger instance (optional)
        **kwargs: Additional training parameters
        
    Returns:
        Dictionary containing training results
    """
    # Load configuration
    config = Config(config_path) if config_path else Config()
    
    # Create unified trainer with provided logger
    trainer = UnifiedTrainer(config, logger)
    
    # Train model
    results = trainer.train_single_model(
        data_path=data_path,
        model_type=model_type,
        output_dir=output_dir,
        **kwargs
    )
    
    return results


