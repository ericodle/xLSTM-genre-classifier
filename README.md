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

# Project requires GTZAN be pre-processed to 30-second clips

# What can you do with this project?

## You can process music data
### GTZAN
### FMA
### Crops songs, weeds out corrupted data, sorts music into train/val/test sets, and summarizes the distribution of your data.

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

## TODO
### integrate new split preprocessing flow for FMA
### Expand module tests
### debug hyperparameter search
### debug fuzzy logic
### debug multimodal models
### debug autoencoders