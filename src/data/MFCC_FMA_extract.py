#!/usr/bin/env python3
"""
FMA MFCC Extraction Script

This script:
1. Loads FMA MP3-to-genre mapping from fma_mp3_genres.json
2. Collects MP3 files from fma-data/fma_medium
3. Splits files into train/val/test sets (deterministic and reproducible)
4. Generates class distribution analysis (histogram + statistics)
5. Copies MP3 files to train/val/test directories
6. Extracts MFCC features for each split
7. Saves training-ready JSON files

This is the primary script for FMA data processing.
"""

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.core.constants import RANDOM_SEED, TEST_SIZE, TRAIN_SIZE, VAL_SIZE

# Audio processing constants
SAMPLE_RATE = 22050
MFCC_COUNT = 13
N_FFT = 2048
HOP_LENGTH = 512
SEG_LENGTH = 30


def extract_mfcc_from_audio(
    audio_path: str,
    mfcc_count: int = MFCC_COUNT,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    seg_length: int = SEG_LENGTH,
) -> Optional[np.ndarray]:
    """
    Extract MFCC features from an audio file.

    Args:
        audio_path: Path to the audio file (MP3)
        mfcc_count: Number of MFCC coefficients
        n_fft: FFT window size
        hop_length: Number of samples between successive frames
        seg_length: Length of audio segment in seconds

    Returns:
        MFCC features as numpy array, or None if extraction fails
    """
    try:
        # Load the audio file
        audio_sig, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    except Exception as e:
        print(f"Error loading file {audio_path}: {e}")
        return None

    # Calculate the number of samples per segment
    seg_samples = seg_length * SAMPLE_RATE

    # Determine the segment to use
    if len(audio_sig) >= seg_samples:
        # Use middle segment for longer files
        middle_index = len(audio_sig) // 2
        segment_start = max(0, middle_index - (seg_samples // 2))
        segment_end = min(len(audio_sig), middle_index + (seg_samples // 2))
        print(f"{audio_path}, segment:{segment_start}-{segment_end} (middle 30s)")
    else:
        # Use the whole file for shorter files
        segment_start = 0
        segment_end = len(audio_sig)
        actual_length = len(audio_sig) / SAMPLE_RATE
        print(f"{audio_path}, using whole file (length: {actual_length:.1f}s)")

    # Extract MFCCs for the segment
    try:
        mfcc = librosa.feature.mfcc(
            y=audio_sig[segment_start:segment_end],
            sr=sr,
            n_mfcc=mfcc_count,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        # Transpose the MFCC matrix
        mfcc = mfcc.T
        return mfcc
    except Exception as e:
        print(f"Error computing MFCCs for {audio_path}: {e}")
        return None


def load_fma_genre_mapping(genre_json_path: str) -> Tuple[Dict[str, str], List[str]]:
    """
    Load FMA MP3-to-genre mapping from JSON file.

    Args:
        genre_json_path: Path to fma_mp3_genres.json

    Returns:
        Tuple of (mp3_genre_mapping dictionary, genre names list)
    """
    print(f"📖 Loading FMA genre mapping from {genre_json_path}...")

    with open(genre_json_path, "r") as f:
        data = json.load(f)

    # Extract the mp3_genre_mapping
    mp3_genre_mapping = data.get("mp3_genre_mapping", {})
    genres = data.get("metadata", {}).get("genres", [])

    print(f"   Loaded {len(mp3_genre_mapping)} MP3-genre mappings")
    print(f"   Found {len(genres)} unique genres: {genres}")

    return mp3_genre_mapping, genres


def collect_mp3_files(
    audio_dir: str, mp3_genre_mapping: Dict[str, str]
) -> List[Tuple[str, str, str]]:
    """
    Collect all MP3 files with their genres from the FMA directory structure.

    Args:
        audio_dir: Directory containing FMA audio files (fma-data/fma_medium)
        mp3_genre_mapping: Dictionary mapping MP3 paths to genres

    Returns:
        List of (file_path, genre, filename) tuples
    """
    print(f"🔍 Collecting MP3 files from {audio_dir}...")

    # Create a normalized mapping dictionary
    # The JSON keys are like "mfccs/fma_medium/042/042279.mp3"
    # We need to match against files like "fma-data/fma_medium/042/042279.mp3"
    normalized_mapping = {}
    for key, genre in mp3_genre_mapping.items():
        # Extract the key part after "fma_medium/"
        # e.g., "mfccs/fma_medium/042/042279.mp3" -> "042/042279.mp3"
        if "fma_medium/" in key:
            normalized_key = key.split("fma_medium/", 1)[1]
            normalized_mapping[normalized_key] = genre

    print(f"   Created normalized mapping with {len(normalized_mapping)} entries")

    audio_files = []

    # Walk through the directory structure
    for root, dirs, files in os.walk(audio_dir):
        for filename in files:
            if filename.endswith(".mp3"):
                full_path = os.path.join(root, filename)

                # Extract relative path from inside fma_medium/
                # e.g., "fma-data/fma_medium/042/042279.mp3" -> "042/042279.mp3"
                relative_from_fma_medium = os.path.relpath(full_path, audio_dir)

                # Try to match with normalized genre mapping
                genre = normalized_mapping.get(relative_from_fma_medium)

                if genre is not None:
                    audio_files.append((full_path, genre, filename))
                    if len(audio_files) % 1000 == 0:
                        print(f"   Matched {len(audio_files)} files...")

    print(f"   Collected {len(audio_files)} MP3 files with genre information")

    return audio_files


def split_files_stratified(
    audio_files: List[Tuple[str, str, str]],
    train_size: float = TRAIN_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_SEED,
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    """
    Split audio files into train/val/test sets with stratification by genre.

    Args:
        audio_files: List of (file_path, genre, filename) tuples
        train_size: Proportion of data for training
        val_size: Proportion of data for validation
        random_state: Random seed for reproducibility

    Returns:
        Three lists: (train_files, val_files, test_files)
    """
    # Separate files by genre
    genre_groups = {}
    for file_path, genre, filename in audio_files:
        if genre not in genre_groups:
            genre_groups[genre] = []
        genre_groups[genre].append((file_path, genre, filename))

    train_files = []
    val_files = []
    test_files = []

    print("\n📊 Splitting files by genre:")

    for genre, files in sorted(genre_groups.items()):
        num_files = len(files)

        # First split: train vs (val + test)
        test_size_split = 1.0 - train_size
        train, temp = train_test_split(
            files, test_size=test_size_split, random_state=random_state, shuffle=True
        )

        # Second split: val vs test
        test_size = 1.0 - train_size - val_size
        val_ratio = val_size / (val_size + test_size)
        val, test = train_test_split(
            temp, test_size=1.0 - val_ratio, random_state=random_state, shuffle=True
        )

        train_files.extend(train)
        val_files.extend(val)
        test_files.extend(test)

        print(
            f"  {genre:20s}: {len(train):4d} train, {len(val):4d} val, {len(test):4d} test (total: {num_files:4d})"
        )

    return train_files, val_files, test_files


def copy_files_to_split(
    files: List[Tuple[str, str, str]], output_dir: str, split_name: str
) -> None:
    """
    Copy files to a split directory, maintaining genre structure.

    Args:
        files: List of (file_path, genre, filename) tuples
        output_dir: Base output directory
        split_name: Name of the split (train/val/test)
    """
    split_dir = os.path.join(output_dir, split_name)
    Path(split_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Copying files to {split_name}/...")

    for source_path, genre, filename in tqdm(files, desc=f"  {split_name}"):
        genre_dir = os.path.join(split_dir, genre)
        Path(genre_dir).mkdir(parents=True, exist_ok=True)

        dest_path = os.path.join(genre_dir, filename)
        shutil.copy2(source_path, dest_path)


def generate_class_distribution_plots(
    train_files: List[Tuple[str, str, str]],
    val_files: List[Tuple[str, str, str]],
    test_files: List[Tuple[str, str, str]],
    genres: List[str],
    output_dir: str,
) -> None:
    """Generate histogram of class distribution across splits."""
    print(f"\n📊 Generating class distribution plots...")

    # Count files by genre for each split
    train_genres = [genre for _, genre, _ in train_files]
    val_genres = [genre for _, genre, _ in val_files]
    test_genres = [genre for _, genre, _ in test_files]

    train_counts = Counter(train_genres)
    val_counts = Counter(val_genres)
    test_counts = Counter(test_genres)

    # Get all genres and ensure ordering
    all_genres = sorted(genres)

    train_values = [train_counts.get(genre, 0) for genre in all_genres]
    val_values = [val_counts.get(genre, 0) for genre in all_genres]
    test_values = [test_counts.get(genre, 0) for genre in all_genres]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(all_genres))
    width = 0.25

    # Create bars
    bars1 = ax.bar(x - width, train_values, width, label="Train", color="#2ecc71", alpha=0.8)
    bars2 = ax.bar(x, val_values, width, label="Val", color="#f39c12", alpha=0.8)
    bars3 = ax.bar(x + width, test_values, width, label="Test", color="#e74c3c", alpha=0.8)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{int(height)}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    # Customize plot
    ax.set_xlabel("Genre", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Files", fontsize=12, fontweight="bold")
    ax.set_title(
        "FMA Dataset: Class Distribution Across Train/Val/Test Splits",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(all_genres, rotation=45, ha="right")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, "class_distribution.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved histogram to {plot_path}")


def save_descriptive_statistics(
    train_files: List[Tuple[str, str, str]],
    val_files: List[Tuple[str, str, str]],
    test_files: List[Tuple[str, str, str]],
    genres: List[str],
    output_dir: str,
) -> None:
    """Save descriptive statistics to a text file."""
    print(f"📝 Saving descriptive statistics...")

    # Count genres
    train_genres = [genre for _, genre, _ in train_files]
    val_genres = [genre for _, genre, _ in val_files]
    test_genres = [genre for _, genre, _ in test_files]

    train_counts = Counter(train_genres)
    val_counts = Counter(val_genres)
    test_counts = Counter(test_genres)

    # Write statistics to file
    stats_path = os.path.join(output_dir, "split_statistics.txt")
    with open(stats_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("FMA Dataset Split Statistics\n")
        f.write("=" * 70 + "\n\n")

        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total files:     {len(train_files) + len(val_files) + len(test_files)}\n")
        f.write(
            f"Train files:     {len(train_files)} ({len(train_files) / (len(train_files) + len(val_files) + len(test_files)) * 100:.1f}%)\n"
        )
        f.write(
            f"Val files:       {len(val_files)} ({len(val_files) / (len(train_files) + len(val_files) + len(test_files)) * 100:.1f}%)\n"
        )
        f.write(
            f"Test files:      {len(test_files)} ({len(test_files) / (len(train_files) + len(val_files) + len(test_files)) * 100:.1f}%)\n"
        )
        f.write(f"Number of genres: {len(genres)}\n")
        f.write(f"Genres:          {', '.join(genres)}\n\n")

        # Per-genre breakdown
        f.write("PER-GENRE BREAKDOWN\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Genre':<20} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8}\n")
        f.write("-" * 70 + "\n")

        for genre in sorted(genres):
            train_count = train_counts.get(genre, 0)
            val_count = val_counts.get(genre, 0)
            test_count = test_counts.get(genre, 0)
            total = train_count + val_count + test_count
            f.write(f"{genre:<20} {train_count:<8} {val_count:<8} {test_count:<8} {total:<8}\n")

    print(f"✅ Saved statistics to {stats_path}")


def extract_mfcc_for_split(
    split_dir: str, output_json: str, genres: List[str], MAX_SAMPLES: int = None
) -> None:
    """
    Extract MFCC features for all files in a split and save to JSON.

    Args:
        split_dir: Directory containing split files (e.g., splits/train/)
        output_json: Path to save the JSON file
        genres: List of all genre names
        MAX_SAMPLES: Maximum number of samples to process (None = all)
    """
    print(f"\n🎵 Extracting MFCC features from {split_dir}...")

    features = []
    labels = []
    file_paths = []
    genre_mapping = {genre: idx for idx, genre in enumerate(genres)}

    # Walk through genre subdirectories
    for genre in genres:
        genre_dir = os.path.join(split_dir, genre)
        if not os.path.exists(genre_dir):
            continue

        genre_files = sorted([f for f in os.listdir(genre_dir) if f.endswith(".mp3")])

        for filename in tqdm(genre_files, desc=f"  {genre}"):
            file_path = os.path.join(genre_dir, filename)

            # Extract MFCC features
            mfcc = extract_mfcc_from_audio(file_path)

            if mfcc is not None:
                features.append(mfcc.tolist())
                labels.append(genre_mapping[genre])
                file_paths.append(file_path)

                if MAX_SAMPLES and len(features) >= MAX_SAMPLES:
                    break

        if MAX_SAMPLES and len(features) >= MAX_SAMPLES:
            break

    # Save to JSON
    output_data = {
        "dataset_type": "fma",
        "split": os.path.basename(split_dir),
        "features": features,
        "labels": labels,
        "mapping": genres,
        "file_paths": file_paths,
    }

    with open(output_json, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"✅ Saved {len(features)} samples to {output_json}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Split FMA data into train/val/test sets and extract MFCCs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline: split files and extract MFCCs
  python src/data/MFCC_FMA_extract.py fma-data/fma_medium src/data/fma_mp3_genres.json fma-data/splits fma-data/mfccs_splits

  Steps performed:
  1. Load MP3-to-genre mapping from JSON
  2. Split files into train/val/test directories (70%/15%/15% default)
  3. Copy MP3 files to respective split directories
  4. Extract MFCC features for each split
  5. Generate class distribution plots and statistics
        """,
    )

    parser.add_argument("audio_dir", help="Path to FMA audio directory (fma-data/fma_medium)")
    parser.add_argument("genre_json", help="Path to fma_mp3_genres.json")
    parser.add_argument("splits_dir", help="Directory to save train/val/test splits")
    parser.add_argument("mfcc_dir", help="Directory to save MFCC JSON files")
    parser.add_argument(
        "--train-size",
        type=float,
        default=TRAIN_SIZE,
        help=f"Proportion for training (default: {TRAIN_SIZE})",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=VAL_SIZE,
        help=f"Proportion for validation (default: {VAL_SIZE})",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})",
    )

    args = parser.parse_args()

    # Step 1: Load genre mapping
    mp3_genre_mapping, genres = load_fma_genre_mapping(args.genre_json)

    # Step 2: Collect MP3 files
    audio_files = collect_mp3_files(args.audio_dir, mp3_genre_mapping)

    if not audio_files:
        print("❌ Error: No MP3 files found with genre information")
        return 1

    # Step 3: Split files
    train_files, val_files, test_files = split_files_stratified(
        audio_files,
        train_size=args.train_size,
        val_size=args.val_size,
        random_state=args.random_state,
    )

    print(f"\n📊 Split summary:")
    print(f"   Train: {len(train_files)} files")
    print(f"   Val:   {len(val_files)} files")
    print(f"   Test:  {len(test_files)} files")

    # Create output directories
    Path(args.splits_dir).mkdir(parents=True, exist_ok=True)
    Path(args.mfcc_dir).mkdir(parents=True, exist_ok=True)

    # Generate statistics and plots
    save_descriptive_statistics(train_files, val_files, test_files, genres, args.splits_dir)
    generate_class_distribution_plots(train_files, val_files, test_files, genres, args.splits_dir)

    # Step 4: Copy files to split directories
    copy_files_to_split(train_files, args.splits_dir, "train")
    copy_files_to_split(val_files, args.splits_dir, "val")
    copy_files_to_split(test_files, args.splits_dir, "test")

    # Step 5: Extract MFCCs for each split
    extract_mfcc_for_split(
        os.path.join(args.splits_dir, "train"), os.path.join(args.mfcc_dir, "train.json"), genres
    )

    extract_mfcc_for_split(
        os.path.join(args.splits_dir, "val"), os.path.join(args.mfcc_dir, "val.json"), genres
    )

    extract_mfcc_for_split(
        os.path.join(args.splits_dir, "test"), os.path.join(args.mfcc_dir, "test.json"), genres
    )

    print(f"\n✅ FMA data processing complete!")
    print(f"\n📁 Output structure:")
    print(f"  {args.splits_dir}/")
    print(f"    ├── train/ (genres: {len(genres)})")
    print(f"    ├── val/   (genres: {len(genres)})")
    print(f"    ├── test/  (genres: {len(genres)})")
    print(f"    ├── class_distribution.png (histogram)")
    print(f"    └── split_statistics.txt (descriptive stats)")
    print(f"  {args.mfcc_dir}/")
    print(f"    ├── train.json")
    print(f"    ├── val.json")
    print(f"    └── test.json")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
