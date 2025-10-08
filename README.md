# GenreDiscern V2 - Refactored Music Genre Classification System

## Current Version Stack
- Python 3.13.0
- PyTorch: 2.8.0+cu128 (compiled with CUDA 12.8)
- NVIDIA Driver Version: 535.247.01
- CUDA Toolkit: 13.0

## Setup (Debian)

- Use pyenv to install python 3.13.0
```bash
pyenv shell 3.13.0
python -m venv env
source env/bin/activate
```

- Install dependencies
```bash
pip install -r requirements.txt
```

## Setup (Windows)

- Use pyenv to install python 3.13.1 (Tcl dependency issue with 3.13.0)
```bash
pyenv shell 3.13.1
python -m venv env
.\env\Scripts\Activate
```

- Install dependencies
```bash
pip install -r requirements.txt
```

## Environment Setup

```bash
source env/bin/activate
```

## Testing

```bash
python run_tests.py
```

## MFCC Configuration

### GTZAN Dataset
```bash
python src/main.py extract --input /path/to/dataset/ --output ./mfccs --name gtzan_13 --n-mfcc 13
```

### FMA Dataset Extraction (takes about 45 minutes)
```bash
python src/MFCC_FMA_extract.py ./mfccs/fma_medium ./mfccs/tracks.csv ./mfccs fma_medium_features --subset medium --mfcc-count 13
```

## Training Commands

### Single Model Training

#### Option 1: Using Main CLI (Recommended)
```bash
# Train CNN Model with automatic evaluation
python src/main.py train --data ./mfccs/gtzan_13.json --model CNN --output ./output/cnn_model

# Train LSTM Model with custom parameters
python src/main.py train --data ./mfccs/gtzan_13.json --model LSTM --output ./output/lstm_model --epochs 50 --batch-size 32

# Train xLSTM Model
python src/main.py train --data ./mfccs/gtzan_13.json --model xLSTM --output ./output/xlstm_model
```

#### Option 2: Using Training Script
```bash
# New unified approach (recommended)
python src/train_model.py --data ./mfccs/gtzan_13.json --model CNN --output ./output/cnn_model --lr 0.001

# Legacy style (still supported)
python src/train_model.py ./mfccs/gtzan_13.json CNN ./output/cnn_model 0.001
```

### Grid Search Hyperparameter Optimization
```bash
# Run grid search for CNN model
python run_grid_search.py --model CNN --data ./mfccs/gtzan_13.json --output ./output/cnn_gridsearch

# Run grid search with custom parameters
python run_grid_search.py --model LSTM --data ./mfccs/gtzan_13.json --output ./output/lstm_gridsearch --params ./src/training/lstm_params.json

# Run grid search with resume capability
python run_grid_search.py --model GRU --data ./mfccs/gtzan_13.json --output ./output/gru_gridsearch --resume
```

### OFAT (One-Factor-at-a-Time) Analysis
```bash
# Run OFAT analysis for CNN model
python run_ofat_analysis.py --model CNN --data ./mfccs/gtzan_13.json --output ./output/cnn_ofat

# Run OFAT analysis for LSTM model
python run_ofat_analysis.py --model LSTM --data ./mfccs/gtzan_13.json --output ./output/lstm_ofat

# Run OFAT analysis with custom config
python run_ofat_analysis.py --model GRU --data ./mfccs/gtzan_13.json --output ./output/gru_ofat --config ./ofat_configs/gru_gtzan_config.json

# Run OFAT analysis for all models
python run_all_ofat.py --data ./mfccs/gtzan_13.json

# Run OFAT analysis for all models with FMA dataset
python run_all_ofat.py --data ./mfccs/fma_13.json

# Run specific models only
python run_all_ofat.py --data ./mfccs/gtzan_13.json --models CNN LSTM GRU
```

### Supported Model Types
- **FC**: Fully Connected Neural Network
- **CNN**: Convolutional Neural Network  
- **LSTM**: Long Short-Term Memory
- **GRU**: Gated Recurrent Unit
- **xLSTM**: Extended LSTM
- **Tr_FC**: Transformer with FC layers
- **Tr_CNN**: Transformer with CNN layers
- **Tr_LSTM**: Transformer with LSTM layers
- **Tr_GRU**: Transformer with GRU layers

## Unified Training System

This project uses a **unified training system** that ensures consistency across all training scenarios:

### ✅ **Consistency**
- Single model training, grid search, and OFAT analysis all use the **same training logic**
- Identical results across all training scenarios
- No more inconsistencies between different training approaches

### ✅ **Maintainability** 
- Single point of maintenance for training logic
- ~800 lines of duplicated code eliminated
- Consistent error handling and logging

### ✅ **Backward Compatibility**
- All existing scripts continue to work
- Legacy interfaces maintained
- Gradual migration path available

## Multi-Dataset Support

### Variable Output Size
The system now automatically detects the number of classes from your dataset and adapts all models accordingly:

- **GTZAN Dataset**: 10 genres (automatically detected)
- **FMA Dataset**: 16 genres (automatically detected)
- **Custom Datasets**: Any number of classes (automatically detected)

### Supported Datasets
```bash
# GTZAN (10 classes)
python src/train_model.py mfccs/gtzan_13.json CNN output_dir 0.001

# FMA (16 classes) 
python src/train_model.py mfccs/fma_13_with_labels.json CNN output_dir 0.001

# Any custom dataset with N classes
python src/train_model.py mfccs/your_dataset.json CNN output_dir 0.001
```

### Model Architecture
All models now support variable output dimensions:
- **FC_model**: `FC_model(num_classes=10)` or `FC_model(num_classes=16)`
- **CNN_model**: `CNN_model(num_classes=10)` or `CNN_model(num_classes=16)`
- **LSTM_model**: `LSTM_model(..., output_dim=10)` or `LSTM_model(..., output_dim=16)`
- **GRU_model**: `GRU_model(..., output_dim=10)` or `GRU_model(..., output_dim=16)`
- **Transformer models**: All support `output_dim` parameter

### Data Format Requirements
Your dataset JSON file must contain:
```json
{
  "features": [[[mfcc_values...], ...], ...],
  "labels": [0, 1, 2, ...]
}
```

The system will automatically:
1. Detect the number of unique classes from the labels
2. Adapt all model architectures to the correct output size
3. Handle variable sequence lengths by padding to max length

## Quick Reference

### Most Common Commands
```bash
# Single model training (recommended)
python src/main.py train --data ./mfccs/gtzan_13.json --model CNN --output ./output/cnn_model

# Grid search
python run_grid_search.py --model CNN --data ./mfccs/gtzan_13.json --output ./output/cnn_gridsearch

# OFAT analysis  
python run_ofat_analysis.py --model CNN --data ./mfccs/gtzan_13.json --output ./output/cnn_ofat
```

### Extract MFCC Features
```bash
# GTZAN dataset
python src/main.py extract --input /path/to/gtzan --output ./mfccs --name gtzan_13 --n-mfcc 13

# FMA dataset
python src/main.py extract --input /path/to/fma --output ./mfccs --name fma_13 --dataset-type fma --fma-api-key YOUR_KEY
```

### Run Tests
```bash
python run_tests.py
```