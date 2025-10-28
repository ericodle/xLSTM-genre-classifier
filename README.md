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

# What can you do with this project?

## You can process music data

### Quick Start: Process GTZAN Data (Single Command)

**One command to process GTZAN data from raw audio to training-ready format:**

```bash
python src/data/split_gtzan_data.py gtzan-data/processed gtzan-data/splits gtzan-data/mfccs_splits
```

This single command:
1. ✅ Collects audio files from `gtzan-data/processed/` (organized by genre)
2. ✅ Splits files into train/val/test sets (70%/15%/15% by default)
3. ✅ Copies WAV files to `gtzan-data/splits/train/`, `splits/val/`, `splits/test/`
4. ✅ Extracts MFCC features for each split
5. ✅ Saves training-ready JSON files: `gtzan-data/mfccs_splits/train.json`, `val.json`, `test.json`

**Then train your model:**
```bash
python src/training/train_model.py --data gtzan-data/mfccs_splits --model GRU --output outputs/my-run
```

### Why Pre-splitting?

- **Reproducibility**: Same splits across all training runs
- **Traceability**: You know exactly which WAV files are in train/val/test
- **Consistency**: All models train on the same data splits for fair comparison
- **Data integrity**: Train/val/test sets never overlap

### FMA
Coming soon...

## You can train a variety of original and commonly-used models
## SVM
### FC
### CNN
### LSTM
### GRU
### xLSTM
### Transformer
### VGG16
### ViT

## You can study the results of that training process with our evaluation and analysis tools
### Tensorboard integation for checking gradients, etc.
### Accuracy and Loss plots during the training routine
### Evaluation confuson matrix and statistics

## You can run a hyperparameter search to make sure you get the best model

## You can explore fuzzy logic, multimodal models, autoencoders, and more.

## You can customize single training runs using in-line flag arguments
### lr
### pretrained
### and more

## You can easily understand how this project is structured
### All the source code is in the src/ folder
### Different subfolders in src/ make up the different modules

## You can test the project components
### Run all tests with: `python test.py`
### Individual test files in `tests/` directory
### Tests include MFCC extraction validation, model functionality, and more

## You can contribute to this project!
### How about making a Pull Request and contributing your own feature?
### How about helping find and fix our many bugs?
