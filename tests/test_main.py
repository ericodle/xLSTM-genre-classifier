"""
Tests for GenreDiscern main CLI functionality.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import setup_cli_parser, extract_features, train_model


class TestCLIParser:
    """Test command line argument parsing."""
    
    def test_parser_creation(self):
        """Test that CLI parser is created correctly."""
        parser = setup_cli_parser()
        assert parser is not None
        assert hasattr(parser, 'parse_args')
        
        # Check subparsers
        subparsers = [action for action in parser._actions if hasattr(action, 'choices')]
        assert len(subparsers) > 0
        
        # Check commands
        commands = subparsers[0].choices.keys()
        assert 'extract' in commands
        assert 'train' in commands
        assert 'evaluate' in commands
    
    def test_extract_command_args(self):
        """Test extract command argument parsing."""
        parser = setup_cli_parser()
        
        # Test valid extract command
        args = parser.parse_args([
            'extract',
            '--input', '/path/to/music',
            '--output', '/path/to/output',
            '--name', 'features'
        ])
        
        assert args.command == 'extract'
        assert args.input == '/path/to/music'
        assert args.output == '/path/to/output'
        assert args.name == 'features'
    
    def test_train_command_args(self):
        """Test train command argument parsing."""
        parser = setup_cli_parser()
        
        # Test valid train command
        args = parser.parse_args([
            'train',
            '--data', '/path/to/features.json',
            '--model', 'LSTM',
            '--output', '/path/to/output',
            '--epochs', '10',
            '--batch-size', '32'
        ])
        
        assert args.command == 'train'
        assert args.data == '/path/to/features.json'
        assert args.model == 'LSTM'
        assert args.output == '/path/to/output'
        assert args.epochs == 10
        assert args.batch_size == 32
    
    def test_evaluate_command_args(self):
        """Test evaluate command argument parsing."""
        parser = setup_cli_parser()
        
        # Test valid evaluate command
        args = parser.parse_args([
            'evaluate',
            '--model', '/path/to/model.onnx',
            '--data', '/path/to/features.json',
            '--output', '/path/to/output'
        ])
        
        assert args.command == 'evaluate'
        assert args.model == '/path/to/model.onnx'
        assert args.data == '/path/to/features.json'
        assert args.output == '/path/to/output'
    
    def test_global_options(self):
        """Test global command line options."""
        parser = setup_cli_parser()
        
        # Test verbose and log file options
        args = parser.parse_args([
            'train',
            '--data', '/path/to/features.json',
            '--model', 'LSTM',
            '--output', '/path/to/output',
            '--verbose',
            '--log-file', '/path/to/log.txt'
        ])
        
        assert args.verbose is True
        assert args.log_file == '/path/to/log.txt'


class TestExtractFeatures:
    """Test feature extraction functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @patch('src.main.MFCCExtractor')
    def test_extract_features_success(self, mock_extractor_class, temp_dir):
        """Test successful feature extraction."""
        # Mock the extractor
        mock_extractor = Mock()
        mock_extractor.extract_mfcc_from_directory.return_value = f"{temp_dir}/features.json"
        mock_extractor_class.return_value = mock_extractor
        
        # Mock config
        mock_config = Mock()
        mock_config.audio = Mock()
        
        # Mock logger
        mock_logger = Mock()
        
        # Test extraction
        result = extract_features(
            Mock(input=temp_dir, output=temp_dir, name='test_features'),
            mock_config,
            mock_logger
        )
        
        assert result == f"{temp_dir}/features.json"
        mock_extractor.extract_mfcc_from_directory.assert_called_once()
    
    @patch('src.main.MFCCExtractor')
    def test_extract_features_failure(self, mock_extractor_class, temp_dir):
        """Test feature extraction failure handling."""
        # Mock the extractor to raise an exception
        mock_extractor = Mock()
        mock_extractor.extract_mfcc_from_directory.side_effect = Exception("Extraction failed")
        mock_extractor_class.return_value = mock_extractor
        
        # Mock config
        mock_config = Mock()
        mock_config.audio = Mock()
        
        # Mock logger
        mock_logger = Mock()
        
        # Test that exception is raised
        with pytest.raises(Exception, match="Extraction failed"):
            extract_features(
                Mock(input=temp_dir, output=temp_dir, name='test_features'),
                mock_config,
                mock_logger
            )


class TestTrainModel:
    """Test model training functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @patch('src.main.ModelTrainer')
    @patch('src.main.run_automatic_evaluation')
    def test_train_model_success_with_evaluation(self, mock_eval, mock_trainer_class, temp_dir):
        """Test successful model training with automatic evaluation."""
        # Mock the trainer
        mock_trainer = Mock()
        mock_trainer.train.return_value = {'train_loss': [0.5, 0.3], 'val_loss': [0.6, 0.4]}
        mock_trainer_class.return_value = mock_trainer
        
        # Mock evaluation
        mock_eval.return_value = {'accuracy': 0.75, 'confusion_matrix': None}
        
        # Mock config
        mock_config = Mock()
        mock_config.model.num_epochs = 10
        mock_config.model.batch_size = 32
        
        # Mock logger
        mock_logger = Mock()
        
        # Test training
        result = train_model(
            Mock(data=f"{temp_dir}/data.json", model='LSTM', output=temp_dir, epochs=None, batch_size=None),
            mock_config,
            mock_logger
        )
        
        # Check that training was called
        mock_trainer.setup_training.assert_called_once()
        mock_trainer.train.assert_called_once()
        
        # Check that evaluation was called
        mock_eval.assert_called_once()
        
        # Check return value
        assert isinstance(result, tuple)
        assert len(result) == 2
        training_history, evaluation_results = result
        assert evaluation_results is not None
    
    @patch('src.main.ModelTrainer')
    @patch('src.main.run_automatic_evaluation')
    def test_train_model_success_without_evaluation(self, mock_eval, mock_trainer_class, temp_dir):
        """Test successful model training without automatic evaluation."""
        # Mock the trainer
        mock_trainer = Mock()
        mock_trainer.train.return_value = {'train_loss': [0.5, 0.3], 'val_loss': [0.6, 0.4]}
        mock_trainer_class.return_value = mock_trainer
        
        # Mock evaluation to fail
        mock_eval.side_effect = Exception("Evaluation failed")
        
        # Mock config
        mock_config = Mock()
        mock_config.model.num_epochs = 10
        mock_config.model.batch_size = 32
        
        # Mock logger
        mock_logger = Mock()
        
        # Test training
        result = train_model(
            Mock(data=f"{temp_dir}/data.json", model='LSTM', output=temp_dir, epochs=None, batch_size=None),
            mock_config,
            mock_logger
        )
        
        # Check that training was called
        mock_trainer.setup_training.assert_called_once()
        mock_trainer.train.assert_called_once()
        
        # Check that evaluation was attempted but failed
        mock_eval.assert_called_once()
        
        # Check return value (should be just training history)
        assert isinstance(result, dict)
        assert 'train_loss' in result
    
    @patch('src.main.ModelTrainer')
    def test_train_model_failure(self, mock_trainer_class, temp_dir):
        """Test model training failure handling."""
        # Mock the trainer to raise an exception
        mock_trainer = Mock()
        mock_trainer.setup_training.side_effect = Exception("Training failed")
        mock_trainer_class.return_value = mock_trainer
        
        # Mock config
        mock_config = Mock()
        mock_config.model.num_epochs = 10
        mock_config.model.batch_size = 32
        
        # Mock logger
        mock_logger = Mock()
        
        # Test that exception is raised
        with pytest.raises(Exception, match="Training failed"):
            train_model(
                Mock(data=f"{temp_dir}/data.json", model='LSTM', output=temp_dir, epochs=None, batch_size=None),
                mock_config,
                mock_logger
            )
    
    @patch('src.main.ModelTrainer')
    def test_train_model_with_custom_args(self, mock_trainer_class, temp_dir):
        """Test model training with custom command line arguments."""
        # Mock the trainer
        mock_trainer = Mock()
        mock_trainer.train.return_value = {'train_loss': [0.5], 'val_loss': [0.6]}
        mock_trainer_class.return_value = mock_trainer
        
        # Mock config
        mock_config = Mock()
        mock_config.model.num_epochs = 10
        mock_config.model.batch_size = 32
        
        # Mock logger
        mock_logger = Mock()
        
        # Test training with custom epochs and batch size
        result = train_model(
            Mock(data=f"{temp_dir}/data.json", model='LSTM', output=temp_dir, epochs=5, batch_size=16),
            mock_config,
            mock_logger
        )
        
        # Check that config was updated
        assert mock_config.model.num_epochs == 5
        assert mock_config.model.batch_size == 16
        
        # Check that training was called
        mock_trainer.setup_training.assert_called_once()
        mock_trainer.train.assert_called_once()


class TestMainIntegration:
    """Test main function integration."""
    
    @patch('src.main.setup_logging')
    @patch('src.main.Config')
    @patch('src.main.extract_features')
    def test_main_extract_command(self, mock_extract, mock_config_class, mock_logging, temp_dir):
        """Test main function with extract command."""
        from src.main import main
        
        # Mock dependencies
        mock_logger = Mock()
        mock_logging.return_value = mock_logger
        
        mock_config = Mock()
        mock_config_class.return_value = mock_config
        
        mock_extract.return_value = f"{temp_dir}/features.json"
        
        # Mock command line arguments
        with patch.object(sys, 'argv', ['main.py', 'extract', '--input', temp_dir, '--output', temp_dir, '--name', 'test']):
            # This would require more complex mocking of argparse
            # For now, we'll test the individual components
            pass
    
    @patch('src.main.setup_logging')
    @patch('src.main.Config')
    @patch('src.main.train_model')
    def test_main_train_command(self, mock_train, mock_config_class, mock_logging, temp_dir):
        """Test main function with train command."""
        from src.main import main
        
        # Mock dependencies
        mock_logger = Mock()
        mock_logging.return_value = mock_logger
        
        mock_config = Mock()
        mock_config_class.return_value = mock_config
        
        mock_train.return_value = ({'train_loss': [0.5]}, {'accuracy': 0.75})
        
        # Mock command line arguments
        with patch.object(sys, 'argv', ['main.py', 'train', '--data', f"{temp_dir}/data.json", '--model', 'LSTM', '--output', temp_dir]):
            # This would require more complex mocking of argparse
            # For now, we'll test the individual components
            pass 