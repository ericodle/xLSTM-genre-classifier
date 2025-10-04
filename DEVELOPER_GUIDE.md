# Developer Guide - Variable Output Size Support

## Overview
All models in this project now support variable output dimensions to work with different datasets (GTZAN: 10 classes, FMA: 16 classes, custom datasets: any number of classes).

## Key Changes Made

### 1. Model Architecture Updates
All models now accept a `num_classes` or `output_dim` parameter:

```python
# Before (hardcoded to 10 classes)
model = CNN_model()

# After (variable classes)
model = CNN_model(num_classes=16)  # For FMA dataset
model = CNN_model(num_classes=10)  # For GTZAN dataset
model = CNN_model(num_classes=25)  # For custom dataset
```

### 2. Automatic Class Detection
The training script automatically detects the number of classes from your dataset:

```python
# In train_model.py main() function
num_classes = len(np.unique(y))
print(f"Number of classes detected: {num_classes}")
```

### 3. Model Instantiation
All models are automatically configured with the correct output dimension:

```python
# All these models now use num_classes variable
if model_type == "FC":
    model = models.FC_model(num_classes=num_classes)
elif model_type == "CNN":
    model = models.CNN_model(num_classes=num_classes)
elif model_type == "LSTM":
    model = models.LSTM_model(..., output_dim=num_classes, ...)
# ... etc for all models
```

## Developer Reminders

### ✅ DO:
- Always pass the `num_classes` parameter when manually instantiating models
- Use the automatic detection in `train_model.py` for training
- Check that your dataset has both "features" and "labels" keys
- Ensure labels are integers starting from 0

### ❌ DON'T:
- Hardcode the number of classes in model definitions
- Assume all datasets have the same number of classes
- Forget to update model instantiation when adding new models

## Supported Models

| Model | Parameter Name | Example |
|-------|----------------|---------|
| FC_model | `num_classes` | `FC_model(num_classes=16)` |
| CNN_model | `num_classes` | `CNN_model(num_classes=16)` |
| LSTM_model | `output_dim` | `LSTM_model(..., output_dim=16, ...)` |
| GRU_model | `output_dim` | `GRU_model(..., output_dim=16, ...)` |
| xLSTM | `num_classes` | `xLSTMClassifier(..., num_classes=16, ...)` |
| Tr_FC | `output_dim` | `Tr_FC(..., output_dim=16, ...)` |
| Tr_CNN | `output_dim` | `Tr_CNN(..., output_dim=16, ...)` |
| Tr_LSTM | `output_dim` | `Tr_LSTM(..., output_dim=16, ...)` |
| Tr_GRU | `output_dim` | `Tr_GRU(..., output_dim=16, ...)` |

## Data Format Requirements

Your dataset JSON must have this structure:
```json
{
  "features": [
    [[mfcc_1_1, mfcc_1_2, ...], [mfcc_2_1, mfcc_2_2, ...], ...],
    [[mfcc_1_1, mfcc_1_2, ...], [mfcc_2_1, mfcc_2_2, ...], ...],
    ...
  ],
  "labels": [0, 1, 2, 0, 1, ...]
}
```

## Testing Your Changes

```bash
# Test with GTZAN (10 classes)
python src/train_model.py mfccs/gtzan_13.json CNN test_output 0.001

# Test with FMA (16 classes)
python src/train_model.py mfccs/fma_13_with_labels.json CNN test_output 0.001

# Test with custom dataset (N classes)
python src/train_model.py mfccs/your_dataset.json CNN test_output 0.001
```

## Adding New Models

When adding new models, remember to:

1. **Accept variable output size**:
```python
class YourNewModel(nn.Module):
    def __init__(self, num_classes=10):  # or output_dim=10
        # ... your model definition
        self.output_layer = nn.Linear(hidden_size, num_classes)
```

2. **Update training script**:
```python
elif model_type == "YourNewModel":
    model = models.YourNewModel(num_classes=num_classes)
```

3. **Add documentation**:
```python
"""
Your new model for music genre classification.

Args:
    num_classes (int): Number of output classes. Defaults to 10 for GTZAN.
                      Use 16 for FMA dataset, or any other number for custom datasets.
"""
```

## Common Issues

### Issue: "Expected all tensors to be on the same device"
**Solution**: Ensure dummy_input for ONNX export is on the same device as the model:
```python
dummy_input = torch.randn(1, X_train.shape[1], X_train.shape[2]).to(device)
```

### Issue: "ValueError: setting an array element with a sequence"
**Solution**: The data loading function now handles variable sequence lengths by padding to max length automatically.

### Issue: "KeyError: 'mfcc'"
**Solution**: Updated to use "features" key instead of "mfcc" for consistency across datasets.
