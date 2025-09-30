"""
Tests for the dataset-agnostic system.
"""

import pytest
import tempfile
import os
import json
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data.datasets.base import BaseDataset
from data.datasets.gtzan import GTZANDataset
from data.datasets.fma import FMADataset
from data.datasets.factory import DatasetFactory


class TestBaseDataset:
    """Test the abstract base dataset class."""

    def test_base_dataset_abstract(self):
        """Test that BaseDataset cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseDataset()

    def test_base_dataset_methods(self):
        """Test that BaseDataset has the required abstract methods."""
        assert hasattr(BaseDataset, "get_audio_files")
        assert hasattr(BaseDataset, "get_genres")
        assert hasattr(BaseDataset, "get_metadata")
        assert hasattr(BaseDataset, "validate_file")
        assert hasattr(BaseDataset, "get_sample_count")
        assert hasattr(BaseDataset, "get_genre_distribution")


class TestGTZANDataset:
    """Test the GTZAN dataset handler."""

    @pytest.fixture
    def temp_gtzan_dir(self):
        """Create a temporary GTZAN-style directory structure."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create genre directories
            genres = [
                "blues",
                "classical",
                "country",
                "disco",
                "hiphop",
                "jazz",
                "metal",
                "pop",
                "reggae",
                "rock",
            ]
            for genre in genres:
                genre_dir = os.path.join(tmp_dir, genre)
                os.makedirs(genre_dir, exist_ok=True)

                # Create some dummy .wav files
                for i in range(3):
                    dummy_file = os.path.join(genre_dir, f"{genre}_{i}.wav")
                    with open(dummy_file, "w") as f:
                        f.write("dummy wav content")

            yield tmp_dir

    def test_gtzan_initialization(self, temp_gtzan_dir):
        """Test GTZAN dataset initialization."""
        dataset = GTZANDataset(temp_gtzan_dir)
        assert dataset.root_dir == temp_gtzan_dir
        assert len(dataset.GENRES) == 10

    def test_gtzan_get_genres(self, temp_gtzan_dir):
        """Test getting genres from GTZAN dataset."""
        dataset = GTZANDataset(temp_gtzan_dir)
        genres = dataset.get_genres()
        assert len(genres) == 10
        assert "blues" in genres
        assert "rock" in genres

    def test_gtzan_get_audio_files(self, temp_gtzan_dir):
        """Test getting audio files from GTZAN dataset."""
        dataset = GTZANDataset(temp_gtzan_dir)
        audio_files = dataset.get_audio_files()

        # Should have 30 files (10 genres * 3 files each)
        assert len(audio_files) == 30

        # Check that all files have correct extensions and genres
        for file_path, genre in audio_files:
            assert file_path.endswith(".wav")
            assert genre in dataset.GENRES
            assert os.path.exists(file_path)

    def test_gtzan_validate_file(self, temp_gtzan_dir):
        """Test file validation for GTZAN dataset."""
        dataset = GTZANDataset(temp_gtzan_dir)

        assert dataset.validate_file("test.wav")
        assert not dataset.validate_file("test.mp3")
        assert not dataset.validate_file("test.txt")

    def test_gtzan_get_metadata(self, temp_gtzan_dir):
        """Test metadata retrieval from GTZAN dataset."""
        dataset = GTZANDataset(temp_gtzan_dir)
        metadata = dataset.get_metadata()

        assert metadata["name"] == "GTZAN"
        assert metadata["genre_count"] == 10
        assert metadata["structure"] == "genre-based folders"
        assert "total_samples" in metadata
        assert "genre_distribution" in metadata

    def test_gtzan_validate_structure(self, temp_gtzan_dir):
        """Test GTZAN structure validation."""
        dataset = GTZANDataset(temp_gtzan_dir)
        assert dataset.validate_structure()

    def test_gtzan_get_genre_directory(self, temp_gtzan_dir):
        """Test getting genre directory paths."""
        dataset = GTZANDataset(temp_gtzan_dir)
        blues_dir = dataset.get_genre_directory("blues")
        assert os.path.exists(blues_dir)
        assert blues_dir == os.path.join(temp_gtzan_dir, "blues")


class TestFMADataset:
    """Test the FMA dataset handler."""

    @pytest.fixture
    def temp_fma_dir(self):
        """Create a temporary FMA-style directory structure."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create track_id-based directories
            for i in range(3):
                track_id = f"{i:06d}"
                subdir = os.path.join(tmp_dir, track_id[:3])
                os.makedirs(subdir, exist_ok=True)

                # Create dummy .mp3 file
                dummy_file = os.path.join(subdir, f"{track_id}.mp3")
                with open(dummy_file, "w") as f:
                    f.write("dummy mp3 content")

            yield tmp_dir

    @pytest.fixture
    def sample_tracks_csv(self):
        """Create a sample tracks CSV file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            # Create a simple tracks CSV structure
            tracks_data = {
                "track_id": [1, 2, 3],
                "track_title": ["Song 1", "Song 2", "Song 3"],
                "track_genre_top": ["Electronic", "Rock", "Jazz"],
                "artist_name": ["Artist 1", "Artist 2", "Artist 3"],
                "album_title": ["Album 1", "Album 2", "Album 3"],
            }
            df = pd.DataFrame(tracks_data)
            df.to_csv(f.name, index=False)
            yield f.name
            os.unlink(f.name)

    def test_fma_initialization(self, temp_fma_dir):
        """Test FMA dataset initialization."""
        dataset = FMADataset(temp_fma_dir, "fake_api_key")
        assert dataset.root_dir == temp_fma_dir
        assert dataset.api_key == "fake_api_key"

    @patch("data.datasets.fma.sys.path")
    def test_fma_load_tracks_data_csv(
        self, mock_sys_path, temp_fma_dir, sample_tracks_csv
    ):
        """Test loading tracks data from CSV."""
        # Mock the FMA utils import
        mock_utils = Mock()
        mock_utils.load.return_value = pd.DataFrame(
            {
                "track_id": [1, 2, 3],
                "track_title": ["Song 1", "Song 2", "Song 3"],
                "track_genre_top": ["Electronic", "Rock", "Jazz"],
            }
        )

        with patch.dict("sys.modules", {"utils": mock_utils}):
            dataset = FMADataset(temp_fma_dir, "fake_api_key", sample_tracks_csv)
            # This should not raise an error since we're mocking the utils
            assert dataset.tracks_csv == sample_tracks_csv

    def test_fma_validate_file(self, temp_fma_dir):
        """Test file validation for FMA dataset."""
        dataset = FMADataset(temp_fma_dir, "fake_api_key")

        assert dataset.validate_file("test.mp3")
        assert not dataset.validate_file("test.wav")
        assert not dataset.validate_file("test.txt")

    def test_fma_get_audio_path_fallback(self, temp_fma_dir):
        """Test FMA audio path generation fallback."""
        dataset = FMADataset(temp_fma_dir, "fake_api_key")

        # Test fallback path generation
        path = dataset._get_audio_path(123)
        expected = os.path.join(temp_fma_dir, "000", "000123.mp3")
        assert path == expected


class TestDatasetFactory:
    """Test the dataset factory."""

    @pytest.fixture
    def temp_gtzan_dir(self):
        """Create a temporary GTZAN-style directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            for genre in [
                "blues",
                "classical",
                "country",
                "disco",
                "hiphop",
                "jazz",
                "metal",
                "pop",
                "reggae",
                "rock",
            ]:
                os.makedirs(os.path.join(tmp_dir, genre), exist_ok=True)
            yield tmp_dir

    @pytest.fixture
    def temp_fma_dir(self):
        """Create a temporary FMA-style directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            for i in range(3):
                subdir = os.path.join(tmp_dir, f"{i:03d}")
                os.makedirs(subdir, exist_ok=True)
            yield tmp_dir

    def test_factory_create_gtzan(self, temp_gtzan_dir):
        """Test creating GTZAN dataset through factory."""
        dataset = DatasetFactory.create_dataset(temp_gtzan_dir, "gtzan")
        assert isinstance(dataset, GTZANDataset)
        assert dataset.root_dir == temp_gtzan_dir

    def test_factory_create_fma(self, temp_fma_dir):
        """Test creating FMA dataset through factory."""
        with patch.dict(os.environ, {"FMA_API_KEY": "test_key"}):
            dataset = DatasetFactory.create_dataset(temp_fma_dir, "fma")
            assert isinstance(dataset, FMADataset)
            assert dataset.root_dir == temp_fma_dir

    def test_factory_auto_detect_gtzan(self, temp_gtzan_dir):
        """Test auto-detection of GTZAN dataset."""
        dataset_type = DatasetFactory._auto_detect_dataset(temp_gtzan_dir)
        assert dataset_type == "gtzan"

    def test_factory_auto_detect_fma(self, temp_fma_dir):
        """Test auto-detection of FMA dataset."""
        dataset_type = DatasetFactory._auto_detect_dataset(temp_fma_dir)
        assert dataset_type == "fma"

    def test_factory_auto_detect_unknown(self, tmpdir):
        """Test auto-detection with unknown structure."""
        # Create a directory with no clear structure
        tmp_dir = str(tmpdir)
        os.makedirs(os.path.join(tmp_dir, "unknown"), exist_ok=True)

        dataset_type = DatasetFactory._auto_detect_dataset(tmp_dir)
        # Should default to GTZAN
        assert dataset_type == "gtzan"

    def test_factory_get_supported_datasets(self):
        """Test getting information about supported datasets."""
        supported = DatasetFactory.get_supported_datasets()

        assert "gtzan" in supported
        assert "fma" in supported
        assert supported["gtzan"]["genres"] == 10
        assert supported["fma"]["genres"] == 16
        assert not supported["gtzan"]["api_required"]
        assert supported["fma"]["api_required"]

    def test_factory_validation(self, temp_gtzan_dir):
        """Test dataset validation through factory."""
        is_valid = DatasetFactory.validate_dataset(temp_gtzan_dir, "gtzan")
        assert is_valid

    def test_factory_validation_invalid_type(self, temp_gtzan_dir):
        """Test validation with invalid dataset type."""
        with pytest.raises(ValueError):
            DatasetFactory.create_dataset(temp_gtzan_dir, "invalid_type")


class TestDatasetIntegration:
    """Integration tests for the dataset system."""

    @pytest.fixture
    def temp_gtzan_dir(self):
        """Create a temporary GTZAN directory with audio files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            for genre in ["blues", "classical"]:
                genre_dir = os.path.join(tmp_dir, genre)
                os.makedirs(genre_dir, exist_ok=True)

                # Create dummy audio files
                for i in range(2):
                    audio_file = os.path.join(genre_dir, f"{genre}_{i}.wav")
                    with open(audio_file, "w") as f:
                        f.write("dummy audio content")

            yield tmp_dir

    def test_dataset_metadata_consistency(self, temp_gtzan_dir):
        """Test that dataset metadata is consistent across methods."""
        dataset = GTZANDataset(temp_gtzan_dir)

        # Get metadata and sample count
        metadata = dataset.get_metadata()
        sample_count = dataset.get_sample_count()
        genre_dist = dataset.get_genre_distribution()

        # Check consistency
        assert metadata["total_samples"] == sample_count
        assert metadata["genre_distribution"] == genre_dist
        assert len(metadata["genres"]) == metadata["genre_count"]

    def test_dataset_file_validation_consistency(self, temp_gtzan_dir):
        """Test that file validation is consistent with actual files."""
        dataset = GTZANDataset(temp_gtzan_dir)
        audio_files = dataset.get_audio_files()

        # All returned files should pass validation
        for file_path, _ in audio_files:
            assert dataset.validate_file(file_path)
            assert os.path.exists(file_path)
