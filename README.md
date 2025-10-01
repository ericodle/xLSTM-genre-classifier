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

- Install CUDA 13.0
```bash
wget https://developer.download.nvidia.com/compute/cuda/13.0.1/local_installers/cuda-repo-debian12-13-0-local_13.0.1-580.82.07-1_amd64.deb
sudo dpkg -i cuda-repo-debian12-13-0-local_13.0.1-580.82.07-1_amd64.deb
sudo cp /var/cuda-repo-debian12-13-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-0
```

- Install dependencies
```bash
pip install -r requirements.txt
pip install -r requirements-test.txt  # For testing
```

## Setup (Windows)

- Use pyenv to install python 3.13.0
```bash
pyenv shell 3.13.0
python -m venv env
source env/bin/activate
```

- Install CUDA 13.0
```bash
# Download and install CUDA 13.0 from NVIDIA website
```

- Install dependencies
```bash
pip install -r requirements.txt
pip install -r requirements-test.txt  # For testing
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

## Training Commands

### Single Model Training
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

### Hyperparameter Optimization

#### Grid Search
```bash
python run_grid_search.py --model CNN --data ./mfccs/gtzan_13.json --output ./output/CNN_run --params ./src/training/cnn_params.json
```

#### OFAT (One-Factor-at-a-Time) Analysis
```bash
# Full OFAT analysis
python run_ofat_analysis.py --model CNN --data ./mfccs/gtzan_13.json --output ./output/ofat_analysis

# Custom OFAT configuration
python run_ofat_analysis.py --model CNN --data ./mfccs/gtzan_13.json --output ./output/ofat_analysis --config ofat_configs/example_custom_config.json

# Specific parameters only
python run_ofat_analysis.py --model CNN --data ./mfccs/gtzan_13.json --output ./output/ofat_analysis --params conv_layers kernel_size dropout
```

## Model Architecture Warning System

The system includes automatic warnings for potentially problematic model configurations:
- **Large model warnings** (>50M parameters)
- **Deep network warnings** (8+ conv layers)
- **Memory usage alerts** for GPU training
- **Parameter sensitivity analysis** via OFAT

## Configuration Files

- **Grid Search**: `src/training/cnn_params.json`
- **OFAT Analysis**: `ofat_configs/cnn_ofat_config.json`
- **Model Parameters**: `src/training/cnn_architecture_params.json`