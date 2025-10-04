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

### FMA Dataset Extraction
```bash
#coming soon
```

## Training Commands

### Single Model Training
```bash
# Train CNN Model
python src/main.py train --data ./mfccs/gtzan_13.json --model CNN --output ./output/cnn_model
```

#### Grid Search
```bash
python run_grid_search.py --model CNN --data ./mfccs/gtzan_13.json --output ./output/CNN_run --params ./src/training/cnn_params.json
```

#### OFAT (One-Factor-at-a-Time) Analysis
```bash
python run_ofat_analysis.py --model CNN --data ./mfccs/gtzan_13.json --output ./output/ofat_analysis
```

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