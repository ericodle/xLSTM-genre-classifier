# OFAT Configuration Files

This directory contains JSON configuration files for One-Factor-at-a-Time (OFAT) parameter sensitivity analysis.

## File Structure

Each model type has its own configuration file:
- `cnn_ofat_config.json` - CNN model OFAT configuration
- `fc_ofat_config.json` - Fully Connected model OFAT configuration  
- `lstm_ofat_config.json` - LSTM model OFAT configuration
- `gru_ofat_config.json` - GRU model OFAT configuration
- `example_custom_config.json` - Example custom configuration

## Configuration Format

Each config file contains three main sections:

### 1. `baseline_config`
The baseline parameter values used when testing other parameters. All parameters except the one being tested will use these values.

### 2. `parameter_ranges`
The range of values to test for each parameter. Each parameter maps to a list of values to test.

### 3. `parameters_to_test`
A list of which parameters to include in the OFAT analysis. If not specified, all parameters in `parameter_ranges` will be tested.

## Usage

### Default Configuration
```bash
# Uses ofat_configs/cnn_ofat_config.json automatically
python run_ofat_analysis.py --model CNN --data ./mfccs/gtzan_13.json --output ./output/ofat_analysis
```

### Custom Configuration
```bash
# Use a custom config file
python run_ofat_analysis.py --model CNN --data ./mfccs/gtzan_13.json --output ./output/ofat_analysis --config my_custom_config.json
```

### Specific Parameters Only
```bash
# Test only specific parameters (overrides config file)
python run_ofat_analysis.py --model CNN --data ./mfccs/gtzan_13.json --output ./output/ofat_analysis --params conv_layers kernel_size
```

## Creating Custom Configurations

1. Copy an existing config file as a starting point
2. Modify the `baseline_config` to set your preferred baseline values
3. Adjust `parameter_ranges` to define what values to test for each parameter
4. Update `parameters_to_test` to specify which parameters to include
5. Use the `--config` argument to specify your custom file

## Example: Quick CNN Test

For a quick test with only 3 parameters and limited ranges:

```json
{
  "baseline_config": {
    "num_classes": 10,
    "conv_layers": 3,
    "base_filters": 32,
    "kernel_size": 5,
    "pool_size": 2,
    "fc_hidden": 128,
    "dropout": 0.1,
    "learning_rate": 0.001,
    "batch_size": 128,
    "weight_decay": 0.001,
    "optimizer": "adam"
  },
  "parameter_ranges": {
    "conv_layers": [2, 3, 4],
    "kernel_size": [3, 5, 7],
    "dropout": [0.1, 0.2, 0.3]
  },
  "parameters_to_test": [
    "conv_layers",
    "kernel_size", 
    "dropout"
  ]
}
```

This would run 3 + 3 + 3 = 9 total training runs instead of the full parameter space.
