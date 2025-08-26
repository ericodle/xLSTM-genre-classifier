# GenreDiscern - Refactored Music Genre Classification System

A comprehensive, well-organized music genre classification system using deep learning with support for multiple neural network architectures.

## 🚀 What's New in the Refactored Version

### ✨ Improved Architecture
- **Modular Design**: Clean separation of concerns with dedicated packages for core, models, data, training, and GUI
- **Configuration Management**: Centralized configuration with JSON support and environment-specific settings
- **Error Handling**: Comprehensive error handling and logging throughout the system
- **Type Hints**: Full type annotations for better code quality and IDE support

### 🧪 Testing Infrastructure
- **Unit Tests**: Comprehensive test suite with pytest
- **Test Coverage**: Aiming for 80%+ code coverage
- **Fixtures**: Reusable test fixtures and mock objects
- **CI/CD Ready**: Structured for continuous integration

### 📦 Better Organization
- **Package Structure**: Proper Python package organization with `__init__.py` files
- **Import Management**: Clean import hierarchy and dependency management
- **Documentation**: Comprehensive docstrings and inline documentation

## 🏗️ Project Structure

```
GenreDiscern/
├── src/                          # Source code
│   ├── __init__.py              # Main package
│   ├── main.py                  # CLI entry point
│   ├── core/                    # Core functionality
│   │   ├── __init__.py
│   │   ├── config.py            # Configuration management
│   │   ├── data_loader.py       # Data loading utilities
│   │   └── utils.py             # Common utilities
│   ├── models/                  # Neural network models
│   │   ├── __init__.py
│   │   ├── base.py              # Base model class
│   │   ├── neural_networks.py   # Traditional NN models
│   │   ├── transformers.py      # Transformer-based models
│   │   └── xlstm.py            # Extended LSTM models
│   ├── data/                    # Data processing
│   │   ├── __init__.py
│   │   ├── mfcc_extractor.py    # MFCC feature extraction
│   │   └── preprocessing.py     # Data preprocessing
│   ├── training/                # Training modules
│   │   ├── __init__.py
│   │   ├── trainer.py           # Main trainer
│   │   ├── evaluator.py         # Model evaluation
│   │   └── grid_search.py       # Hyperparameter optimization
│   └── gui/                     # Graphical interface
│       ├── __init__.py
│       ├── main_window.py       # Main GUI window
│       └── windows/             # Individual window modules
├── tests/                       # Test suite
│   ├── __init__.py
│   ├── conftest.py             # Pytest configuration
│   ├── test_core/              # Core module tests
│   ├── test_models/            # Model tests
│   ├── test_data/              # Data processing tests
│   ├── test_training/          # Training tests
│   └── test_gui/               # GUI tests
├── requirements.txt             # Main dependencies
├── requirements-test.txt        # Testing dependencies
├── pytest.ini                 # Pytest configuration
└── README_REFACTORED.md        # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/ericodle/GenreDiscern
cd GenreDiscern

# Create virtual environment
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-test.txt
```

### 2. Command Line Usage

#### Extract MFCC Features
```bash
python src/main.py extract \
    --input /path/to/music/directory \
    --output /path/to/output \
    --name features
```

#### Train a Model
```bash
python src/main.py train \
    --data /path/to/features.json \
    --model LSTM \
    --output /path/to/output \
    --epochs 100 \
    --batch-size 32
```

#### Launch GUI
```bash
python src/main.py gui
```

### 3. Configuration

Create a custom configuration file `config.json`:

```json
{
  "audio": {
    "sample_rate": 22050,
    "mfcc_count": 13,
    "n_fft": 2048,
    "hop_length": 512
  },
  "model": {
    "batch_size": 64,
    "hidden_size": 128,
    "learning_rate": 0.001,
    "num_epochs": 100
  },
  "training": {
    "device": "auto",
    "num_workers": 4,
    "save_checkpoints": true
  }
}
```

Use with:
```bash
python src/main.py train --config config.json --data data.json --model CNN
```

## 🧪 Running Tests

### Run All Tests
```bash
pytest
```

### Run with Coverage
```bash
pytest --cov=src --cov-report=html
```

### Run Specific Test Categories
```bash
# Unit tests only
pytest -m unit

# Integration tests only
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

### Test Coverage Report
After running tests with coverage, open `htmlcov/index.html` in your browser to view the detailed coverage report.

## 🔧 Development

### Code Quality Tools

```bash
# Format code with Black
black src/ tests/

# Sort imports
isort src/ tests/

# Lint with flake8
flake8 src/ tests/

# Type checking with mypy
mypy src/
```

### Adding New Models

1. Create a new model class in `src/models/` that inherits from `BaseModel`
2. Implement the required `forward` method
3. Add the model to the factory function in `src/models/__init__.py`
4. Write comprehensive tests in `tests/test_models/`

### Adding New Features

1. Follow the existing package structure
2. Add proper type hints and docstrings
3. Write unit tests for new functionality
4. Update documentation

## 📊 Supported Models

### Traditional Neural Networks
- **FC_model**: Fully Connected Neural Network
- **CNN_model**: Convolutional Neural Network
- **LSTM_model**: Long Short-Term Memory
- **GRU_model**: Gated Recurrent Unit

### Advanced Models
- **xLSTM**: Extended LSTM with causal convolutions
- **Transformer-based**: Various transformer architectures

## 🎯 Key Features

- **Modular Architecture**: Easy to extend and maintain
- **Configuration Management**: Flexible configuration system
- **Comprehensive Testing**: Full test suite with high coverage
- **Error Handling**: Robust error handling and logging
- **Type Safety**: Full type annotations throughout
- **Documentation**: Comprehensive inline documentation
- **CLI Interface**: Command-line interface for automation
- **GUI Interface**: User-friendly graphical interface
- **Model Persistence**: Save/load trained models
- **Training Visualization**: Automatic plot generation
- **Early Stopping**: Intelligent training termination
- **Checkpointing**: Resume training from checkpoints

## 🔍 Monitoring and Logging

The system provides comprehensive logging and monitoring:

- **Training Progress**: Real-time training metrics
- **Performance Plots**: Automatic generation of loss, accuracy, and learning rate plots
- **Model Checkpoints**: Regular model saving for recovery
- **Validation Metrics**: Continuous validation performance tracking

## 🚀 Performance Optimization

- **GPU Support**: Automatic CUDA detection and utilization
- **Data Loading**: Optimized data loading with multiple workers
- **Memory Management**: Efficient memory usage and cleanup
- **Batch Processing**: Configurable batch sizes for optimal performance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following the coding standards
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Coding Standards

- Follow PEP 8 style guidelines
- Use type hints for all function parameters and return values
- Write comprehensive docstrings
- Maintain test coverage above 80%
- Use meaningful variable and function names

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Eric and Rebecca**: Original project creators
- **PyTorch Team**: Deep learning framework
- **Librosa Team**: Audio processing library
- **Open Source Community**: Various supporting libraries

## 📞 Support

For questions, issues, or contributions:

1. Check the existing [Issues](../../issues)
2. Create a new issue with detailed description
3. Contact the maintainers

---

**Note**: This is the refactored version of GenreDiscern. For the original version, see the main branch or original repository. 