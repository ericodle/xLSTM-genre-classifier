# GenreDiscern Development Notes

## Environment Setup
```bash
source env/bin/activate
```

## Install Requirements
```bash
pip install -r requirements.txt
pip install -r requirements-test.txt  # For testing
```

### Run Tests
```bash
python run_tests.py
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

## FMA Dataset Extraction

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

## Hyperparameter Grid Search

```bash
python run_grid_search.py --model CNN --data ./mfccs/gtzan_13.json --output ./output/CNN_run --params ./src/training/cnn_params.json
```