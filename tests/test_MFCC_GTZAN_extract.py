#!/usr/bin/env python3
"""
Test MFCC_GTZAN_extract.py script

This test literally calls the exact command users run:
    python src/data/MFCC_GTZAN_extract.py tests/test-gtzan outputs/test-mfcc-extraction/splits outputs/test-mfcc-extraction/mfccs_splits
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestMFCCExtraction:
    """Test MFCC extraction using the actual production command."""

    @pytest.fixture
    def test_data_dir(self):
        """Path to test GTZAN data."""
        return os.path.join(os.path.dirname(__file__), "test-gtzan")

    @pytest.fixture
    def output_dir(self):
        """Output directory for test results."""
        output_path = "outputs/test-mfcc-extraction"
        Path(output_path).mkdir(parents=True, exist_ok=True)
        return output_path

    def test_data_directory_exists(self, test_data_dir):
        """Verify test data directory exists."""
        assert os.path.exists(test_data_dir), f"Test data directory not found: {test_data_dir}"

        # Count files
        file_count = 0
        for genre_dir in os.listdir(test_data_dir):
            genre_path = os.path.join(test_data_dir, genre_dir)
            if os.path.isdir(genre_path):
                files = [f for f in os.listdir(genre_path) if f.endswith(".wav")]
                file_count += len(files)

        assert file_count > 0, f"No audio files found (found {file_count} files)"
        print(f"Found {file_count} audio files in test data")

    def test_run_production_command(self, test_data_dir, output_dir):
        """Run the exact command users would run."""
        # Build the command
        splits_dir = os.path.join(output_dir, "splits")
        mfcc_dir = os.path.join(output_dir, "mfccs_splits")

        cmd = [
            sys.executable,
            "src/data/MFCC_GTZAN_extract.py",
            test_data_dir,
            splits_dir,
            mfcc_dir,
            "--train-size",
            "0.6",
            "--val-size",
            "0.2",
        ]

        print(f"Running: {' '.join(cmd)}")

        # Run the actual production script
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__))
        )

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        assert result.returncode == 0, f"Command failed with return code {result.returncode}"

        # Verify output structure
        assert os.path.exists(splits_dir), "Splits directory should be created"
        assert os.path.exists(mfcc_dir), "MFCC directory should be created"

        for split_name in ["train", "val", "test"]:
            split_json = os.path.join(mfcc_dir, f"{split_name}.json")
            assert os.path.exists(split_json), f"{split_name}.json should exist"

            # Verify JSON structure
            with open(split_json, "r") as f:
                data = json.load(f)

            required_fields = [
                "dataset_type",
                "split",
                "features",
                "labels",
                "mapping",
                "file_paths",
            ]
            for field in required_fields:
                assert field in data, f"{split_name}.json should have {field} field"

            assert len(data["features"]) > 0, f"{split_name} should have features"
            print(f"✅ {split_name}.json: {len(data['features'])} samples")


if __name__ == "__main__":
    # Allow running as a script for quick testing
    pytest.main([__file__, "-v", "-s"])
