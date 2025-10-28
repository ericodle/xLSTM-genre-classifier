# Data Processing Module

This module handles all audio data processing and feature extraction for the genre classification project.

## Standard Workflow: Pre-split Method

**All data extractions should use the pre-split method for consistency.**

### GTZAN Dataset

Use `MFCC_GTZAN_extract.py` as the **primary tool** for GTZAN data processing:

```bash
python src/data/MFCC_GTZAN_extract.py gtzan-data/processed gtzan-data/splits gtzan-data/mfccs_splits
```

This single command:
1. Collects audio files from `gtzan-data/processed/` (organized by genre)
2. Splits files into train/val/test sets (70%/15%/15% default)
3. Generates `class_distribution.png` histogram and `split_statistics.txt` descriptive stats
4. Copies WAV files to `gtzan-data/splits/train/`, `splits/val/`, `splits/test/`
5. Extracts MFCC features for each split
6. Saves training-ready JSON files: `gtzan-data/mfccs_splits/train.json`, `val.json`, `test.json`

**Output structure:**
```
gtzan-data/
├── splits/
│   ├── train/
│   │   ├── blues/
│   │   ├── classical/
│   │   └── ...
│   ├── val/
│   ├── test/
│   ├── class_distribution.png (histogram)
│   └── split_statistics.txt (descriptive stats)
└── mfccs_splits/
    ├── train.json
    ├── val.json
    └── test.json
```

### FMA Dataset

Use `MFCC_FMA_extract.py` as the **primary tool** for FMA data processing:

```bash
python src/data/MFCC_FMA_extract.py fma-data/fma_medium src/data/fma_mp3_genres.json fma-data/splits fma-data/mfccs_splits
```

This command:
1. Loads MP3-to-genre mapping from `src/data/fma_mp3_genres.json`
2. Collects MP3 files from `fma-data/fma_medium/` (numerical directory structure)
3. Splits files into train/val/test sets (70%/15%/15% default)
4. Generates `class_distribution.png` histogram and `split_statistics.txt` descriptive stats
5. Copies MP3 files to `fma-data/splits/train/`, `splits/val/`, `splits/test/`
6. Extracts MFCC features for each split
7. Saves training-ready JSON files: `fma-data/mfccs_splits/train.json`, `val.json`, `test.json`

**Note**: FMA uses MP3 files (not WAV like GTZAN) and requires the genre mapping JSON file because genre information is not in the directory structure.

### Why Pre-split?

- **Reproducibility**: Same splits across all training runs
- **Traceability**: You know exactly which WAV files are in train/val/test
- **Consistency**: All models train on the same data splits for fair comparison
- **Data integrity**: Train/val/test sets never overlap

## Files Overview

### Production Scripts

- **`MFCC_GTZAN_extract.py`** ⭐ - **Primary script** for GTZAN data processing with pre-split workflow
  - What users run: `python src/data/MFCC_GTZAN_extract.py gtzan-data/processed gtzan-data/splits gtzan-data/mfccs_splits`

- **`MFCC_FMA_extract.py`** ⭐ - **Primary script** for FMA data processing with pre-split workflow
  - What users run: `python src/data/MFCC_FMA_extract.py fma-data/fma_medium src/data/fma_mp3_genres.json fma-data/splits fma-data/mfccs_splits`

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

1. **Always use `MFCC_GTZAN_extract.py`** for GTZAN data
2. **Always use `MFCC_FMA_extract.py`** for FMA data
3. **Never split data during training** - splits should be fixed before training starts
4. **Keep audio files organized** - Maintain genre subdirectories in splits
5. **Document splits** - Keep track of which files are in which split for reproducibility

