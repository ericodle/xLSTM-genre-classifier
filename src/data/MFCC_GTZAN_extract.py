#!/usr/bin/env python3
"""
GTZAN MFCC Extraction Script

This script:
1. Collects audio files from GTZAN dataset (organized by genre)
2. Splits files into train/val/test sets (deterministic and reproducible)
3. Generates class distribution analysis (histogram + statistics)
4. Copies WAV files to train/val/test directories
5. Extracts MFCC features for each split
6. Saves training-ready JSON files

This is the primary script for GTZAN data processing.
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
import soundfile as sf
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
        audio_path: Path to the audio file
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


def get_gtzan_genres(processed_dir: str) -> List[str]:
    """
    Get GTZAN genre names dynamically from directory structure.

    Args:
        processed_dir: Directory containing processed audio files organized by genre

    Returns:
        Sorted list of genre names
    """
    genres = []
    if os.path.exists(processed_dir):
        # Look for genre directories
        for item in os.listdir(processed_dir):
            item_path = os.path.join(processed_dir, item)
            # Check if it's a directory (genre folder) and contains .wav files
            if os.path.isdir(item_path):
                wav_files = [f for f in os.listdir(item_path) if f.endswith(".wav")]
                if wav_files:
                    genres.append(item)
    return sorted(genres)


def collect_audio_files(processed_dir: str) -> Tuple[List[Tuple[str, str, str]], List[str]]:
    """
    Collect all audio files with their genres and full paths.

    Args:
        processed_dir: Directory containing processed audio files organized by genre

    Returns:
        Tuple of (list of (file_path, genre, filename) tuples, list of genre names)
    """
    # Get genres dynamically from directory structure
    genres = get_gtzan_genres(processed_dir)
    if not genres:
        print(f"⚠️  Warning: No genre directories found in {processed_dir}")
        return [], []

    print(f"📂 Found {len(genres)} genres: {', '.join(genres)}")

    audio_files = []

    for genre in genres:
        genre_dir = os.path.join(processed_dir, genre)
        if os.path.exists(genre_dir):
            files = sorted([f for f in os.listdir(genre_dir) if f.endswith(".wav")])
            for filename in files:
                file_path = os.path.join(genre_dir, filename)
                audio_files.append((file_path, genre, filename))

    return audio_files, genres


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
        # Calculate correct ratio: val_size and test_size should each be 15% of total
        test_size = 1.0 - train_size - val_size
        val_ratio = val_size / (val_size + test_size)
        val, test = train_test_split(
            temp, test_size=1.0 - val_ratio, random_state=random_state, shuffle=True
        )

        train_files.extend(train)
        val_files.extend(val)
        test_files.extend(test)

        print(
            f"  {genre:12s}: {len(train):3d} train, {len(val):3d} val, {len(test):3d} test (total: {num_files:3d})"
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

    # Use provided genres and ensure ordering
    all_genres = sorted(genres)

    train_values = [train_counts.get(genre, 0) for genre in all_genres]
    val_values = [val_counts.get(genre, 0) for genre in all_genres]
    test_values = [test_counts.get(genre, 0) for genre in all_genres]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

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
                    fontsize=9,
                )

    # Customize plot
    ax.set_xlabel("Genre", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Files", fontsize=12, fontweight="bold")
    ax.set_title("Class Distribution Across Train/Val/Test Splits", fontsize=14, fontweight="bold")
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
        f.write("GTZAN Dataset Split Statistics\n")
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
        f.write(f"Genres:          {', '.join(sorted(genres))}\n\n")

        # Per-genre breakdown
        f.write("PER-GENRE BREAKDOWN\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Genre':<15} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8}\n")
        f.write("-" * 70 + "\n")

        for genre in sorted(genres):
            train_count = train_counts.get(genre, 0)
            val_count = val_counts.get(genre, 0)
            test_count = test_counts.get(genre, 0)
            total = train_count + val_count + test_count
            f.write(f"{genre:<15} {train_count:<8} {val_count:<8} {test_count:<8} {total:<8}\n")

    print(f"✅ Saved statistics to {stats_path}")


def extract_mfcc_for_split(
    split_dir: str, output_json: str, genres: List[str], MAX_SAMPLES: int = None
) -> None:
    """
    Extract MFCC features for all files in a split and save to JSON.

    Args:
        split_dir: Directory containing the split (with genre subdirectories)
        output_json: Path to output JSON file
        genres: List of genre names
        MAX_SAMPLES: Optional limit on number of samples to process (for testing)
    """
    print(f"\n📊 Extracting MFCC features for {split_dir}...")

    # Get all files
    audio_files = []
    for genre in genres:
        genre_dir = os.path.join(split_dir, genre)
        if os.path.exists(genre_dir):
            files = sorted([f for f in os.listdir(genre_dir) if f.endswith(".wav")])
            for filename in files:
                file_path = os.path.join(genre_dir, filename)
                audio_files.append((file_path, genre))

    if MAX_SAMPLES:
        audio_files = audio_files[:MAX_SAMPLES]

    # Extract MFCCs
    features = []
    labels = []
    file_mapping = []  # Track which file each feature came from
    label_map = {genre: idx for idx, genre in enumerate(genres)}

    for file_path, genre in tqdm(audio_files, desc="  Extracting"):
        mfcc = extract_mfcc_from_audio(
            file_path,
            mfcc_count=MFCC_COUNT,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            seg_length=SEG_LENGTH,
        )

        if mfcc is not None:
            features.append(mfcc.tolist())
            labels.append(label_map[genre])
            file_mapping.append(os.path.relpath(file_path, split_dir))

    # Save to JSON
    output_data = {
        "dataset_type": "gtzan",
        "split": os.path.basename(split_dir),
        "features": features,
        "labels": labels,
        "mapping": genres,
        "file_paths": file_mapping,  # Include file paths for traceability
    }

    with open(output_json, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✅ Saved {len(features)} samples to {output_json}")
    print(
        f"   Shape per sample: {len(features[0])} frames × {len(features[0][0])} MFCC coefficients"
    )


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Split GTZAN data into train/val/test sets and extract MFCCs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline: split files and extract MFCCs
  python src/data/MFCC_GTZAN_extract.py gtzan-data/processed gtzan-data/splits gtzan-data/mfccs

  Steps performed:
  1. Split files into train/val/test directories (70%/15%/15% default)
  2. Copy WAV files to respective split directories
  3. Extract MFCC features for each split
  4. Save separate JSON files for each split (train.json, val.json, test.json)
        """,
    )

    parser.add_argument("input_dir", help="Directory containing processed GTZAN audio files")
    parser.add_argument(
        "output_dir", help="Directory to create split subdirectories (train/val/test)"
    )
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
    parser.add_argument(
        "--skip-copy", action="store_true", help="Skip copying files (assumes split already exists)"
    )
    parser.add_argument(
        "--skip-mfcc",
        action="store_true",
        help="Skip MFCC extraction (only perform file splitting)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples per split (for testing)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not os.path.exists(args.input_dir):
        print(f"❌ Error: Input directory does not exist: {args.input_dir}")
        return 1

    if abs(args.train_size + args.val_size + (1 - args.train_size - args.val_size) - 1.0) > 1e-6:
        print(f"❌ Error: train_size + val_size must be <= 1.0")
        print(f"   Got: train_size={args.train_size}, val_size={args.val_size}")
        return 1

    print("🎵 GTZAN Data Split and MFCC Extraction")
    print("=" * 60)
    print(f"Input directory:  {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"MFCC directory:   {args.mfcc_dir}")
    print(
        f"Split ratios:     train={args.train_size:.1%}, val={args.val_size:.1%}, test={1-args.train_size-args.val_size:.1%}"
    )
    print()

    # Step 1: Collect audio files
    print("📁 Collecting audio files...")
    audio_files, genres = collect_audio_files(args.input_dir)
    if not audio_files:
        print(f"❌ Error: No audio files found in {args.input_dir}")
        return 1
    print(f"   Found {len(audio_files)} files across {len(genres)} genres")

    # Step 2: Split files
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

    # Create output directory before generating statistics and plots
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Generate statistics and plots
    save_descriptive_statistics(train_files, val_files, test_files, genres, args.output_dir)
    generate_class_distribution_plots(train_files, val_files, test_files, genres, args.output_dir)

    # Step 3: Copy files to split directories
    if not args.skip_copy:
        copy_files_to_split(train_files, args.output_dir, "train")
        copy_files_to_split(val_files, args.output_dir, "val")
        copy_files_to_split(test_files, args.output_dir, "test")
    else:
        print("\n⏭️  Skipping file copy (assumes splits already exist)")

    # Step 4: Extract MFCCs for each split
    if not args.skip_mfcc:
        Path(args.mfcc_dir).mkdir(parents=True, exist_ok=True)

        for split_name in ["train", "val", "test"]:
            split_dir = os.path.join(args.output_dir, split_name)
            output_json = os.path.join(args.mfcc_dir, f"{split_name}.json")

            if os.path.exists(split_dir):
                extract_mfcc_for_split(split_dir, output_json, genres, args.max_samples)
            else:
                print(f"\n⚠️  Split directory not found: {split_dir}")
    else:
        print("\n⏭️  Skipping MFCC extraction")

    print("\n🎉 Processing completed successfully!")
    print(f"\nDirectory structure:")
    print(f"  {args.output_dir}/")
    print(f"    ├── train/  (genres: blues, classical, ...)")
    print(f"    ├── val/    (genres: blues, classical, ...)")
    print(f"    └── test/   (genres: blues, classical, ...)")
    print(f"\n  {args.mfcc_dir}/")
    print(f"    ├── train.json")
    print(f"    ├── val.json")
    print(f"    └── test.json")
    print(f"\n  {args.output_dir}/")
    print(f"    ├── class_distribution.png (histogram)")
    print(f"    └── split_statistics.txt (descriptive stats)")

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
