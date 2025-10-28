# Music Genre Research Project (MGRP)

## About this Project

We are researchers in the Lin Lab at FCU.
We love to study music genre classification.

## Setup
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

## Project requirements
- Music files shoudl be trimmed to the middle 30-seconds.
- Versions of Python, PyTorch, NVIDIA drivers, and CUDA Toolkit that play well together
Our setup uses:
```
- Python 3.13.0
- PyTorch: 2.8.0+cu128 (compiled with CUDA 12.8)
- NVIDIA Driver Version: 535.247.01
- CUDA Toolkit: 13.0
```

# What can you do with this project?

## You can process music data
### GTZAN
```
python src/data/MFCC_GTZAN_extract.py gtzan-data/processed gtzan-data/splits gtzan-data/mfccs_splits
```

### FMA
```
python src/data/MFCC_FMA_extract.py fma-data/fma_medium src/data/fma_mp3_genres.json fma-data/splits fma-data/mfccs_splits
```


# You can train a variety of models with our default settings in `src/core/constants.py`

### SVM
```
some code here
```

### FC
```
some code here
```

### CNN
```
some code here
```

### LSTM
```
some code here
```

### GRU
```
some code here
```

### xLSTM
```
some code here
```

### Transformer
```
some code here
```

### VGG16
```
some code here
```

### ViT
```
some code here
```

# You can customize single training runs using in-line arguments
### lr
```
some code here
```
### pretrained
```
some code here
```
### and more
```
some code here
```


# You can study the results of your training process with our evaluation and analysis tools
### Tensorboard integation for checking gradients, etc.
### Accuracy and Loss plots during the training routine
### Evaluation confuson matrix and statistics

## You can test the features of this project
### Run all tests with: `python test.py`
### Individual test files in `tests/` directory

## You can contribute to this project!
### How about making a Pull Request (PR) and contributing your own feature?
### How about helping us find and fix some bugs?

# Maintainer:
@ericodle https://github.com/ericodle

# TODO
### generate test set for FMA and add FMA preprocessing test
### Add more tests
### Make new feature: src/hyperparameter_search
### Make new feature: src/fuzzy_logic
### Make new feature: src/multimodal_features
### Make new feature: src/GAN_augmentation
### Make new feature: src/NAS