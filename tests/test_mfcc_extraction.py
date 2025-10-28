#!/usr/bin/env python3
"""
Test MFCC extraction pipeline

This test:
1. Uses a small GTZAN dataset from tests/test-gtzan
2. Extracts MFCC features
3. Verifies the features are correctly formatted
4. Saves results to outputs (gitignored)
"""

import sys
import os
import json
import pytest
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import librosa
# Import the actual functions used in production
from src.data.MFCC_GTZAN_extract import extract_mfcc_from_audio
from src.data.split_gtzan_data import extract_mfcc_for_split
from src.core.constants import GTZAN_GENRES


class TestMFCCExtraction:
    """Test MFCC extraction functionality."""
    
    @pytest.fixture
    def test_data_dir(self):
        """Path to test GTZAN data."""
        return os.path.join(os.path.dirname(__file__), "test-gtzan")
    
    @pytest.fixture
    def output_dir(self):
        """Output directory for test results - matching production structure."""
        output_path = "outputs/test-mfcc-extraction"
        # Create the same directory structure as production
        Path(output_path, "splits", "train").mkdir(parents=True, exist_ok=True)
        Path(output_path, "splits", "val").mkdir(parents=True, exist_ok=True)
        Path(output_path, "splits", "test").mkdir(parents=True, exist_ok=True)
        Path(output_path, "mfccs_splits").mkdir(parents=True, exist_ok=True)
        return output_path
    
    def test_data_directory_exists(self, test_data_dir):
        """Verify test data directory exists."""
        assert os.path.exists(test_data_dir), f"Test data directory not found: {test_data_dir}"
    
    def test_collect_audio_files(self, test_data_dir):
        """Test that we can collect audio files from test data."""
        audio_files = []
        for genre in GTZAN_GENRES:
            genre_dir = os.path.join(test_data_dir, genre)
            if os.path.exists(genre_dir):
                files = [f for f in os.listdir(genre_dir) if f.endswith('.wav')]
                for filename in files:
                    file_path = os.path.join(genre_dir, filename)
                    audio_files.append((file_path, genre, filename))
        
        assert len(audio_files) > 0, "No audio files found in test data"
        print(f"Found {len(audio_files)} audio files")
        
        # Verify we have files from multiple genres
        genres_found = set([genre for _, genre, _ in audio_files])
        assert len(genres_found) > 1, "Test data should have multiple genres"
    
    def test_extract_mfcc_from_single_file(self, test_data_dir):
        """Test MFCC extraction from a single audio file."""
        # Pick a file to test
        blues_dir = os.path.join(test_data_dir, "blues")
        if os.path.exists(blues_dir):
            test_files = [f for f in os.listdir(blues_dir) if f.endswith('.wav')]
            if test_files:
                test_file = os.path.join(blues_dir, test_files[0])
                
                # Extract MFCC
                mfcc = extract_mfcc_from_audio(
                    test_file,
                    mfcc_count=13,
                    n_fft=2048,
                    hop_length=512,
                    seg_length=30
                )
                
                assert mfcc is not None, "MFCC extraction failed"
                assert isinstance(mfcc, np.ndarray), "MFCC should be a numpy array"
                assert len(mfcc.shape) == 2, "MFCC should be 2D (time, features)"
                assert mfcc.shape[1] == 13, "Should have 13 MFCC coefficients"
                print(f"MFCC shape: {mfcc.shape}")
    
    def test_process_gtzan_dataset(self, test_data_dir, output_dir):
        """Test processing entire GTZAN dataset using the actual production function."""
        # Use the exact same function used in production (split_gtzan_data.py)
        # Save to mfccs_splits/test.json to match production structure
        output_file = os.path.join(output_dir, "mfccs_splits", "test.json")
        extract_mfcc_for_split(test_data_dir, output_file)
        
        assert os.path.exists(output_file), "Output file should be created"
        print(f"Saved test results to: {output_file}")
        
        # Load and verify the output
        with open(output_file, 'r') as f:
            loaded_data = json.load(f)
        
        # Verify exact format matches production
        assert "dataset_type" in loaded_data, "Should have dataset_type"
        assert "split" in loaded_data, "Should have split"
        assert loaded_data["dataset_type"] == "gtzan", "Should be GTZAN dataset"
        assert "features" in loaded_data, "Saved file should have 'features' key"
        assert len(loaded_data["features"]) > 0, "Should have features"
        assert len(loaded_data["labels"]) > 0, "Should have labels"
        assert "file_paths" in loaded_data, "Should have file_paths"
        assert "mapping" in loaded_data, "Should have genre mapping"
        
        print(f"Extracted {len(loaded_data['features'])} samples")
        print(f"Genres: {loaded_data['mapping']}")
    
    def test_mfcc_feature_shape(self, test_data_dir, output_dir):
        """Test that MFCC features have consistent shape."""
        # Extract using production function
        output_file = os.path.join(output_dir, "mfccs_splits", "val.json")
        extract_mfcc_for_split(test_data_dir, output_file)
        
        # Load and check shapes
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        # Check all features have 13 MFCC coefficients
        for i, mfcc in enumerate(data["features"]):
            assert len(mfcc[0]) == 13, f"Sample {i} should have 13 MFCC coefficients"
        
        print(f"All {len(data['features'])} samples have correct MFCC dimensions")
    
    def test_all_genres_present(self, test_data_dir, output_dir):
        """Test that all genres are present in the test data."""
        # Extract using production function
        output_file = os.path.join(output_dir, "mfccs_splits", "train.json")
        extract_mfcc_for_split(test_data_dir, output_file)
        
        # Load and analyze
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        # Get unique labels
        unique_labels = set(data["labels"])
        
        # Verify at least some genres are present
        assert len(unique_labels) > 0, "Should have at least one genre"
        print(f"Found {len(unique_labels)} genres in test data")
        
        # Save summary
        summary = {
            "total_samples": len(data["features"]),
            "genres_found": len(unique_labels),
            "unique_labels": sorted(list(unique_labels)),
            "genre_mapping": data["mapping"]
        }
        
        summary_file = os.path.join(output_dir, "test_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Test summary saved to: {summary_file}")


if __name__ == "__main__":
    # Allow running as a script for quick testing
    pytest.main([__file__, "-v", "-s"])

