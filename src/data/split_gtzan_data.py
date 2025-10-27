#!/usr/bin/env python3
"""
GTZAN Data Split Script

This script splits GTZAN data into train/val/test sets before processing, ensuring that:
1. The split is deterministic and reproducible
2. Files are organized into train/val/test directories
3. Separate MFCC extraction can be performed for each split
4. The mapping between WAV files and splits is preserved
"""

import sys
import os
import json
import shutil
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import librosa
import soundfile as sf

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.MFCC_GTZAN_extract import extract_mfcc_from_audio
from src.core.constants import TRAIN_SIZE, VAL_SIZE, TEST_SIZE, RANDOM_SEED, GTZAN_GENRES

# Audio processing constants
SAMPLE_RATE = 22050
MFCC_COUNT = 13
N_FFT = 2048
HOP_LENGTH = 512
SEG_LENGTH = 30


def collect_audio_files(processed_dir: str) -> List[Tuple[str, str, str]]:
    """
    Collect all audio files with their genres and full paths.
    
    Args:
        processed_dir: Directory containing processed audio files organized by genre
        
    Returns:
        List of (file_path, genre, filename) tuples
    """
    audio_files = []
    
    for genre in GTZAN_GENRES:
        genre_dir = os.path.join(processed_dir, genre)
        if os.path.exists(genre_dir):
            files = sorted([f for f in os.listdir(genre_dir) if f.endswith('.wav')])
            for filename in files:
                file_path = os.path.join(genre_dir, filename)
                audio_files.append((file_path, genre, filename))
    
    return audio_files


def split_files_stratified(
    audio_files: List[Tuple[str, str, str]], 
    train_size: float = TRAIN_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_SEED
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
            files,
            test_size=test_size_split,
            random_state=random_state,
            shuffle=True
        )
        
        # Second split: val vs test
        # Calculate correct ratio: val_size and test_size should each be 15% of total
        test_size = 1.0 - train_size - val_size
        val_ratio = val_size / (val_size + test_size)
        val, test = train_test_split(
            temp,
            test_size=1.0 - val_ratio,
            random_state=random_state,
            shuffle=True
        )
        
        train_files.extend(train)
        val_files.extend(val)
        test_files.extend(test)
        
        print(f"  {genre:12s}: {len(train):3d} train, {len(val):3d} val, {len(test):3d} test (total: {num_files:3d})")
    
    return train_files, val_files, test_files


def copy_files_to_split(
    files: List[Tuple[str, str, str]], 
    output_dir: str, 
    split_name: str
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


def extract_mfcc_for_split(
    split_dir: str, 
    output_json: str, 
    MAX_SAMPLES: int = None
) -> None:
    """
    Extract MFCC features for all files in a split and save to JSON.
    
    Args:
        split_dir: Directory containing the split (with genre subdirectories)
        output_json: Path to output JSON file
        MAX_SAMPLES: Optional limit on number of samples to process (for testing)
    """
    print(f"\n📊 Extracting MFCC features for {split_dir}...")
    
    # Get all files
    audio_files = []
    for genre in GTZAN_GENRES:
        genre_dir = os.path.join(split_dir, genre)
        if os.path.exists(genre_dir):
            files = sorted([f for f in os.listdir(genre_dir) if f.endswith('.wav')])
            for filename in files:
                file_path = os.path.join(genre_dir, filename)
                audio_files.append((file_path, genre))
    
    if MAX_SAMPLES:
        audio_files = audio_files[:MAX_SAMPLES]
    
    # Extract MFCCs
    features = []
    labels = []
    file_mapping = []  # Track which file each feature came from
    label_map = {genre: idx for idx, genre in enumerate(GTZAN_GENRES)}
    
    for file_path, genre in tqdm(audio_files, desc="  Extracting"):
        mfcc = extract_mfcc_from_audio(
            file_path, 
            mfcc_count=MFCC_COUNT,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            seg_length=SEG_LENGTH
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
        "mapping": GTZAN_GENRES,
        "file_paths": file_mapping  # Include file paths for traceability
    }
    
    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Saved {len(features)} samples to {output_json}")
    print(f"   Shape per sample: {len(features[0])} frames × {len(features[0][0])} MFCC coefficients")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Split GTZAN data into train/val/test sets and extract MFCCs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline: split files and extract MFCCs
  python src/data/split_gtzan_data.py gtzan-data/processed gtzan-data/splits gtzan-data/mfccs

  Steps performed:
  1. Split files into train/val/test directories (70%/15%/15% default)
  2. Copy WAV files to respective split directories
  3. Extract MFCC features for each split
  4. Save separate JSON files for each split (train.json, val.json, test.json)
        """
    )
    
    parser.add_argument("input_dir", help="Directory containing processed GTZAN audio files")
    parser.add_argument("output_dir", help="Directory to create split subdirectories (train/val/test)")
    parser.add_argument("mfcc_dir", help="Directory to save MFCC JSON files")
    
    parser.add_argument("--train-size", type=float, default=TRAIN_SIZE, 
                       help=f"Proportion for training (default: {TRAIN_SIZE})")
    parser.add_argument("--val-size", type=float, default=VAL_SIZE,
                       help=f"Proportion for validation (default: {VAL_SIZE})")
    parser.add_argument("--random-state", type=int, default=RANDOM_SEED,
                       help=f"Random seed (default: {RANDOM_SEED})")
    parser.add_argument("--skip-copy", action="store_true",
                       help="Skip copying files (assumes split already exists)")
    parser.add_argument("--skip-mfcc", action="store_true",
                       help="Skip MFCC extraction (only perform file splitting)")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Limit number of samples per split (for testing)")
    
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
    print(f"Split ratios:     train={args.train_size:.1%}, val={args.val_size:.1%}, test={1-args.train_size-args.val_size:.1%}")
    print()
    
    # Step 1: Collect audio files
    print("📁 Collecting audio files...")
    audio_files = collect_audio_files(args.input_dir)
    print(f"   Found {len(audio_files)} files")
    
    # Step 2: Split files
    train_files, val_files, test_files = split_files_stratified(
        audio_files,
        train_size=args.train_size,
        val_size=args.val_size,
        random_state=args.random_state
    )
    
    print(f"\n📊 Split summary:")
    print(f"   Train: {len(train_files)} files")
    print(f"   Val:   {len(val_files)} files")
    print(f"   Test:  {len(test_files)} files")
    
    # Step 3: Copy files to split directories
    if not args.skip_copy:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
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
                extract_mfcc_for_split(split_dir, output_json, args.max_samples)
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
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

