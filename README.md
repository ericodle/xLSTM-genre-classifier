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