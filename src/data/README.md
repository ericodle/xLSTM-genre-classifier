# Data Processing Module

This module handles all audio data processing and feature extraction for the genre classification project.

## Standard Workflow: Pre-split Method

**All data extractions should use the pre-split method for consistency.**

### GTZAN Dataset

Use `split_gtzan_data.py` as the **primary tool** for GTZAN data processing:

```bash
python src/data/split_gtzan_data.py gtzan-data/processed gtzan-data/splits gtzan-data/mfccs_splits
```

This single command:
1. Collects audio files from `gtzan-data/processed/` (organized by genre)
2. Splits files into train/val/test sets (70%/15%/15% default)
3. Copies WAV files to `gtzan-data/splits/train/`, `splits/val/`, `splits/test/`
4. Extracts MFCC features for each split
5. Saves training-ready JSON files: `gtzan-data/mfccs_splits/train.json`, `val.json`, `test.json`

**Output structure:**
```
gtzan-data/
├── splits/
│   ├── train/
│   │   ├── blues/
│   │   ├── classical/
│   │   └── ...
│   ├── val/
│   └── test/
└── mfccs_splits/
    ├── train.json
    ├── val.json
    └── test.json
```

### Why Pre-split?

- **Reproducibility**: Same splits across all training runs
- **Traceability**: You know exactly which WAV files are in train/val/test
- **Consistency**: All models train on the same data splits for fair comparison
- **Data integrity**: Train/val/test sets never overlap

## Files Overview

### Production Scripts

- **`split_gtzan_data.py`** ⭐ - **Primary script** for GTZAN data processing with pre-split workflow
  - What users run: `python src/data/split_gtzan_data.py gtzan-data/processed gtzan-data/splits gtzan-data/mfccs_splits`

- **`MFCC_FMA_extract.py`** - FMA dataset MFCC extraction (when using FMA dataset)

### Core Functions (Used by Production Scripts)

- **`MFCC_GTZAN_extract.py`** - Core MFCC extraction functions (used by split_gtzan_data.py)
  - `extract_mfcc_from_audio()` - Extract MFCC features from a single audio file
  - `process_gtzan_dataset()` - Process entire GTZAN dataset
  - `save_gtzan_data()` - Save extracted features to JSON
  - ⚠️ Not meant to be called directly - use split_gtzan_data.py instead

### Helper Modules

None - preprocessing functionality is now integrated directly into `src/training/trainer.py`

## Standard Development Pattern

When adding new data processing:

1. **Use pre-split structure**: Always create train/val/test splits before extraction
2. **Save by split**: Create separate files for train.json, val.json, test.json
3. **Include metadata**: Add dataset_type, split name, file paths to JSON output
4. **Test with `tests/test_mfcc_extraction.py`**: Ensure new processing methods work correctly

## Training Integration

The training pipeline (`src/training/trainer.py`) expects pre-split data:

```python
# Trains expect data_path to be a directory with train.json, val.json, test.json
trainer = ModelTrainer(config, logger)
trainer.train(data_path="gtzan-data/mfccs_splits")
```

## Best Practices

1. **Always use `split_gtzan_data.py`** for GTZAN data
2. **Never split data during training** - splits should be fixed before training starts
3. **Keep WAV files organized** - Maintain genre subdirectories in splits
4. **Validate data** - Use `AudioPreprocessor.validate_data()` to check data quality
5. **Document splits** - Keep track of which files are in which split for reproducibility

