# Tests Directory

This directory contains tests for the genre classification project.

## Test Structure

### `test-gtzan/`
Contains a small subset of the GTZAN dataset with 3 songs per genre (27 files total) for testing MFCC extraction and related functionality without needing the full dataset.

Genres included:
- blues
- classical  
- country
- disco
- hiphop
- jazz
- metal
- pop
- reggae
- rock

### Test Files

#### `test_mfcc_extraction.py`
Tests the MFCC extraction pipeline using the test GTZAN dataset.

**Run tests:**
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_mfcc_extraction.py

# Run with verbose output
pytest tests/test_mfcc_extraction.py -v

# Run with output displayed
pytest tests/test_mfcc_extraction.py -s
```

**Test results:**
- Test results are saved to `outputs/test-mfcc-extraction/` (gitignored)
- Features are extracted and validated for shape and content
- JSON output is tested for correct format

## Writing New Tests

When adding new tests:

1. Create test files following the `test_*.py` naming convention
2. Use pytest fixtures for common setup (data directories, paths, etc.)
3. Save test outputs to `outputs/` directory (gitignored)
4. Include docstrings explaining what each test validates
5. Use descriptive test names that indicate what is being tested

## Requirements

Tests require:
- pytest
- All project dependencies (see `requirements.txt`)

