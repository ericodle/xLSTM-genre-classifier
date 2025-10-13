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

### Extract MFCC Features
Use the unified extractor API:
```bash
python -c "from src.core.config import AudioConfig; from src.data.mfcc_extractor import MFCCExtractor; import json; \
cfg=AudioConfig(); ex=MFCCExtractor(cfg); print(ex.extract_mfcc_from_directory('/path/to/music','./mfccs','dataset_13'))"
```




### Most Common Commands
```bash
python src/train_model.py --data mfccs/gtzan_13.json --model GRU --output outputs/gru-gtzan-run
python src/training/train_svm.py --data mfccs/gtzan_13.json --kernel rbf --C 10 --gamma scale --output outputs/svm-gtzan

tensorboard --logdir outputs --port 6006

netron outputs/your_run/model.onnx

```