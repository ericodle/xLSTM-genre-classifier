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

## Quick Reference
### Extract MFCC Features
```bash
# GTZAN dataset
python src/MFCC_GTZAN_extract.py /path/to/gtzan ./mfccs gtzan_13 --n-mfcc 13
# FMA dataset
python src/MFCC_FMA_extract.py /path/to/fma /path/to/tracks.csv ./mfccs fma_13 --subset medium --mfcc-count 13
```
### Run Tests
```bash
python run_tests.py
```








### Most Common Commands
```bash
python src/train_model.py --data mfccs/gtzan_13.json --model GRU --output outputs/gru-gtzan-run

tensorboard --logdir outputs --port 6006

netron outputs/your_run/model.onnx
```