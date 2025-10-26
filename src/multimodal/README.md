# Multimodal Music Genre Classification

This module implements a multimodal approach to music genre classification using specialized neural network branches for different types of audio features.

## Architecture Overview

The multimodal system consists of three specialized branches:

### 1. **Spectral CNN Branch** (`SpectralCNNBranch`)
- **Purpose**: Processes spectral features (mel-spectrogram, chroma, spectral characteristics)
- **Input**: 2D spectral representations
- **Architecture**: Convolutional layers + BatchNorm + ReLU + MaxPooling
- **Features**: 
  - Mel-spectrogram (brightness/timbre)
  - Chroma features (harmonic content)
  - Spectral centroid (brightness)
  - Spectral rolloff (frequency distribution)
  - Spectral contrast (peaks vs valleys)
  - Zero crossing rate (percussiveness)

### 2. **Temporal RNN Branch** (`TemporalRNNBranch`)
- **Purpose**: Processes temporal patterns in audio
- **Input**: Sequential temporal features
- **Architecture**: RNN/LSTM/GRU + Fully connected layers
- **Features**:
  - MFCC coefficients (timbre)
  - Delta MFCC (velocity)
  - Delta-delta MFCC (acceleration)

### 3. **Statistical FC Branch** (`StatisticalFCBranch`)
- **Purpose**: Processes statistical measures and global characteristics
- **Input**: Flattened statistical features
- **Architecture**: Fully connected layers + BatchNorm + Dropout
- **Features**:
  - Tempo (rhythm speed)
  - Beat frames (rhythm patterns)
  - Onset strength (rhythm intensity)
  - Harmonic/percussive ratio (melodic vs rhythmic content)
  - Spectral bandwidth (frequency spread)
  - Spectral flatness (noisiness)

## Fusion Mechanisms

### 1. **Attention Fusion** (`AttentionFusion`)
- Learns to weight different branches based on their relevance
- Uses attention mechanism to combine branch outputs
- Most sophisticated approach

### 2. **Concatenation Fusion** (`ConcatFusion`)
- Simple concatenation of branch outputs
- Passes through fully connected layers
- Straightforward approach

### 3. **Weighted Fusion**
- Simple weighted average of branch outputs
- Learnable weights for each branch
- Lightweight approach

## Usage

### 1. Extract Multimodal Features

```bash
# From audio files
python src/multimodal/extract_multimodal_features.py \
    --input /path/to/audio/directory \
    --output multimodal_features.json \
    --mode audio

# Convert existing MFCC data (dummy features)
python src/multimodal/extract_multimodal_features.py \
    --input mfccs/gtzan_13.json \
    --output multimodal_features.json \
    --mode mfcc
```

### 2. Train Multimodal Model

```bash
python src/multimodal/train_multimodal.py \
    --data multimodal_features.json \
    --output outputs/multimodal_model \
    --fusion attention \
    --epochs 100 \
    --batch-size 32
```

### 3. Programmatic Usage

```python
from multimodal import MultimodalFeatureExtractor, MultimodalModel

# Extract features
extractor = MultimodalFeatureExtractor()
features = extractor.extract_features("audio_file.wav")

# Create model
model = MultimodalModel(
    fusion_method="attention",
    num_classes=10
)

# Forward pass
output = model(features)
```

## Key Benefits

1. **Specialized Processing**: Each branch is optimized for specific feature types
2. **Comprehensive Features**: Captures spectral, temporal, and statistical aspects
3. **Flexible Fusion**: Multiple fusion strategies to choose from
4. **Interpretability**: Can analyze individual branch contributions
5. **Scalability**: Easy to add new branches or fusion methods

## File Structure

```
src/multimodal/
├── __init__.py                    # Module exports
├── feature_extractor.py          # Comprehensive feature extraction
├── model_branches.py             # Specialized neural network branches
├── multimodal_model.py           # Main multimodal model and fusion
├── train_multimodal.py           # Training script
└── extract_multimodal_features.py # Feature extraction utility
```

## Performance Expectations

The multimodal approach should provide:
- **Better accuracy** than single-modal approaches
- **More robust** classification across different music styles
- **Better generalization** due to diverse feature representations
- **Interpretable results** through branch analysis

## Dependencies

- `librosa`: Audio feature extraction
- `torch`: Neural network implementation
- `numpy`: Numerical computations
- `scikit-learn`: Data processing and metrics
- `matplotlib/seaborn`: Visualization

## Notes

- For real multimodal features, you need original audio files
- The MFCC conversion mode creates dummy features for demonstration
- Each branch can be trained independently for analysis
- Attention weights can be visualized to understand branch importance
