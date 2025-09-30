# GenreDiscern Development Notes

## Project Overview

GenreDiscern is a comprehensive music genre classification system that supports multiple neural network architectures, dataset-agnostic MFCC extraction, and automated hyperparameter optimization. The system has been designed with modularity, configurability, and maintainability in mind.

## Quick Commands

### Environment Setup
```bash
source env/bin/activate
```

### Install Requirements
```bash
pip install -r requirements.txt
pip install -r requirements-test.txt  # For testing
```

### Run Tests
```bash
python -m pytest tests/ -v
python run_tests.py  # Alternative test runner
```

## MFCC Configuration

```bash
# High-resolution extraction
python src/main.py extract --input /home/eo/Documents/gtzan/ --output ./mfccs --name gtzan_32 \
    --n-mfcc 32

# Typical extraction 
python src/main.py extract --input /home/eo/Documents/gtzan/ --output ./mfccs --name gtzan_13 \
    --n-mfcc 13

```

### FMA Dataset Extraction

```bash
# Using environment variable for API key
export FMA_API_KEY="your_api_key_here"
python src/main.py extract --input /path/to/fma_dataset --output ./mfccs/ --name fma_mfcc --dataset-type fma

# Using command line argument for API key
python src/main.py extract --input /path/to/fma_dataset --output ./mfccs/ --name fma_mfcc \
    --dataset-type fma --fma-api-key "your_key"

# With tracks CSV for faster processing (recommended)
python src/main.py extract --input /path/to/fma_dataset --output ./mfccs/ --name fma_mfcc \
    --dataset-type fma --fma-tracks-csv /path/to/tracks.csv

# Custom MFCC count for FMA
python src/main.py extract --input /path/to/fma_dataset --output ./mfccs/ --name fma_mfcc \
    --dataset-type fma --fma-api-key "your_key" --n-mfcc 24
```

## Model Training

### Available Model Types

- **FC (Fully Connected)**: Traditional feedforward networks
- **CNN (Convolutional)**: 2D convolutional networks for spatial features
- **LSTM**: Long Short-Term Memory for sequential data
- **GRU**: Gated Recurrent Unit (simplified LSTM)
- **xLSTM**: Extended LSTM with advanced architecture
- **Transformer Variants**: `Tr_FC`, `Tr_CNN`, `Tr_LSTM`, `Tr_GRU`

### Training Commands

```bash
# Train FC Model (automatically evaluates after training)
python src/main.py train --data ./mfccs/gtzan_13.json --model FC --output ./output/fc_model

# Train CNN Model (automatically evaluates after training)
python src/main.py train --data ./mfccs/gtzan_13.json --model CNN --output ./output/cnn_model

# Train LSTM Model (automatically evaluates after training)
python src/main.py train --data ./mfccs/gtzan_13.json --model LSTM --output ./output/lstm_model

# Train GRU Model (automatically evaluates after training)
python src/main.py train --data ./mfccs/gtzan_13.json --model GRU --output ./output/gru_model

# Train xLSTM Model (automatically evaluates after training)
python src/main.py train --data ./mfccs/gtzan_13.json --model xLSTM --output ./output/xlstm_model




# Train with CSV file (pre-extracted MFCC features)
python src/main.py train --data /home/eo/Documents/FMA_full.csv --model FC --output ./output/csv_fc_model
```

### Training Output

Each training run generates:
- `best_model.onnx` - ONNX model (primary format for deployment and evaluation)
- `best_model_metadata.json` - Model configuration and training history
- `best_model_training_metadata.json` - Training-specific metadata (epochs, config, etc.)
- `training_plots/` - Training visualization plots
- `evaluation_results/` - Comprehensive evaluation metrics

**Note**: The system now uses ONNX as the primary model format for better cross-platform compatibility and deployment flexibility.

## Hyperparameter Grid Search

### Basic Usage

```bash
# Run GRU grid search with default parameters
python run_grid_search.py --model GRU --data ./mfccs/gtzan_mfcc.json --output ./output/gru_gridsearch

# Run LSTM grid search with custom parameters
python run_grid_search.py --model LSTM --data ./mfccs/gtzan_mfcc.json --output ./output/lstm_gridsearch \
    --params lstm_params.json

# Run CNN grid search with dry-run to see what would be tested
python run_grid_search.py --model CNN --data ./mfccs/gtzan_mfcc.json --output ./output/cnn_gridsearch --dry-run

# Run xLSTM grid search with verbose logging
python run_grid_search.py --model xLSTM --data ./mfccs/gtzan_mfcc.json --output ./output/xlstm_gridsearch --verbose
```

### Custom Parameter Files

Create JSON files like `gru_params.json`:
```json
{
    "hidden_size": [32, 64, 128],
    "num_layers": [1, 2],
    "dropout": [0.1, 0.2, 0.3],
    "learning_rate": [0.001, 0.01],
    "batch_size": [16, 32]
}
```

### Quick Parameter File Creation

```bash
# Create a custom GRU parameter file
echo '{
    "hidden_size": [64, 128],
    "num_layers": [2, 3],
    "dropout": [0.2, 0.3],
    "learning_rate": [0.001, 0.005],
    "batch_size": [32, 64]
}' > custom_gru_params.json

# Use it in grid search
python run_grid_search.py --model GRU --data ./mfccs/gtzan_mfcc.json \
    --output ./output/custom_gru_search --params custom_gru_params.json
```
