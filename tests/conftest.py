"""
Pytest configuration and fixtures for GenreDiscern tests.
"""

import pytest
import tempfile
import os
import json
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock
import torch

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.config import Config
from data.preprocessing import AudioPreprocessor


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    config = Mock(spec=Config)
    
    # Mock model configuration
    config.model = Mock()
    config.model.hidden_size = 64
    config.model.num_layers = 2
    config.model.dropout = 0.2
    config.model.learning_rate = 0.001
    config.model.batch_size = 32
    config.model.num_filters = 64
    config.model.kernel_size = 3
    config.model.hidden_sizes = [128, 64]
    
    # Mock training configuration
    config.training = Mock()
    config.training.epochs = 10
    config.training.patience = 5
    config.training.random_seed = 42
    config.training.validation_split = 0.2
    config.training.test_split = 0.1
    
    # Mock data configuration
    config.data = Mock()
    config.data.sample_rate = 22050
    config.data.n_mfcc = 13
    config.data.hop_length = 512
    config.data.n_fft = 2048
    
    return config


@pytest.fixture
def sample_mfcc_data():
    """Sample MFCC data for testing."""
    # Create realistic MFCC data
    n_samples = 100
    n_mfcc = 13
    n_frames = 50
    
    features = np.random.randn(n_samples, n_frames, n_mfcc).astype(np.float32)
    labels = np.random.choice(['blues', 'classical', 'country', 'disco', 'hiphop'], n_samples)
    
    return {
        'features': features,
        'labels': labels,
        'metadata': {
            'sample_rate': 22050,
            'n_mfcc': n_mfcc,
            'hop_length': 512,
            'n_fft': 2048
        }
    }


@pytest.fixture
def sample_features():
    """Sample features array for testing."""
    return np.random.randn(100, 50, 13).astype(np.float32)


@pytest.fixture
def sample_labels():
    """Sample labels for testing."""
    return ['blues', 'classical', 'country', 'disco', 'hiphop'] * 20


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def sample_training_history():
    """Sample training history for testing."""
    return {
        'train_losses': [0.8, 0.6, 0.4, 0.3, 0.25],
        'val_losses': [0.9, 0.7, 0.5, 0.4, 0.35],
        'train_accuracies': [0.6, 0.8, 0.9, 0.92, 0.94],
        'val_accuracies': [0.5, 0.7, 0.85, 0.88, 0.90],
        'learning_rates': [0.001, 0.001, 0.0005, 0.0005, 0.0001]
    }


@pytest.fixture
def sample_evaluation_results():
    """Sample evaluation results for testing."""
    return {
        'accuracy': 0.85,
        'precision': 0.87,
        'recall': 0.85,
        'f1_score': 0.86,
        'confusion_matrix': np.array([[15, 2, 1], [1, 18, 1], [2, 1, 17]]),
        'classification_report': 'Sample classification report',
        'roc_auc': 0.92,
        'ks_values': {'blues': 0.15, 'classical': 0.12, 'country': 0.18}
    }


@pytest.fixture
def mock_onnx_session():
    """Mock ONNX runtime session for testing."""
    session = Mock()
    session.run.return_value = [np.random.randn(1, 10)]  # Mock output
    session.get_inputs.return_value = [Mock(name='input', shape=[1, 50, 13])]
    session.get_outputs.return_value = [Mock(name='output', shape=[1, 10])]
    return session


@pytest.fixture
def sample_model_checkpoint():
    """Sample model checkpoint for testing."""
    return {
        'model_state_dict': {
            'layer1.weight': torch.randn(64, 13),
            'layer1.bias': torch.randn(64),
            'layer2.weight': torch.randn(10, 64),
            'layer2.bias': torch.randn(10)
        },
        'optimizer_state_dict': {},
        'epoch': 10,
        'best_val_loss': 0.35,
        'best_val_acc': 0.90
    }


@pytest.fixture
def sample_param_grid():
    """Sample parameter grid for grid search testing."""
    return {
        'hidden_size': [32, 64, 128],
        'num_layers': [1, 2],
        'dropout': [0.1, 0.2, 0.3],
        'learning_rate': [0.001, 0.01],
        'batch_size': [16, 32]
    }


@pytest.fixture
def sample_grid_search_results():
    """Sample grid search results for testing."""
    return pd.DataFrame([
        {
            'combination_id': 0,
            'hidden_size': 32,
            'num_layers': 1,
            'dropout': 0.1,
            'learning_rate': 0.001,
            'batch_size': 16,
            'status': 'completed',
            'best_val_acc': 0.75,
            'best_val_loss': 0.45,
            'output_dir': '/tmp/test1'
        },
        {
            'combination_id': 1,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'learning_rate': 0.01,
            'batch_size': 32,
            'status': 'completed',
            'best_val_acc': 0.82,
            'best_val_loss': 0.38,
            'output_dir': '/tmp/test2'
        }
    ])


@pytest.fixture
def mock_grid_search_trainer():
    """Mock grid search trainer for testing."""
    trainer = Mock()
    trainer.results = []
    trainer.config = Mock()
    trainer.logger = Mock()
    return trainer 