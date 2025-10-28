#!/usr/bin/env python3
"""
GTZAN Audio Processing Script

⚠️  DEPRECATED: Use `split_gtzan_data.py` instead for the standard pre-split workflow.

This script processes GTZAN audio files by:
1. Reading WAV files from the original dataset location
2. Cutting 30-second clips (from the middle of each file)
3. Saving them to gtzan-data/processed with the same subdirectory structure
4. Running MFCC extraction on the processed files and saving to gtzan-data/mfccs

For production use, see: `src/data/split_gtzan_data.py`
"""

import sys
import os
import argparse
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.MFCC_GTZAN_extract import (
    process_gtzan_dataset,
    save_gtzan_data,
    SAMPLE_RATE,
    SONG_LENGTH,
)

########################################################################
# CONSTANT VARIABLES
########################################################################

# Constants for audio processing
SEGMENT_LENGTH = 30  # Duration of each clip in seconds

########################################################################
# AUDIO PROCESSING FUNCTIONS
########################################################################


def cut_30_second_clip(audio_path: str, output_path: str) -> bool:
    """
    Cut a 30-second clip from the middle of an audio file and save it.

    Args:
        audio_path: Path to the input audio file
        output_path: Path to save the output clip

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load the audio file
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

        # Calculate segment length in samples
        segment_samples = SEGMENT_LENGTH * SAMPLE_RATE

        # Determine the segment to use
        if len(audio) >= segment_samples:
            # Use middle segment for longer files
            middle_index = len(audio) // 2
            segment_start = max(0, middle_index - (segment_samples // 2))
            segment_end = min(len(audio), middle_index + (segment_samples // 2))
            audio_clip = audio[segment_start:segment_end]
        else:
            # Use the whole file for shorter files and pad
            audio_clip = audio
            padding = segment_samples - len(audio_clip)
            if padding > 0:
                audio_clip = np.pad(audio_clip, (0, padding), mode="constant")

        # Save the clip
        sf.write(output_path, audio_clip, SAMPLE_RATE)

        return True

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return False


def copy_and_cut_files(
    source_dir: str, dest_dir: str, maintain_structure: bool = True
) -> Dict[str, int]:
    """
    Copy WAV files from source to destination, cutting them to 30-second clips.

    Args:
        source_dir: Source directory containing GTZAN dataset
        dest_dir: Destination directory (gtzan-data/processed)
        maintain_structure: Whether to maintain the subdirectory structure

    Returns:
        Dictionary with statistics about the operation
    """
    print(f"🎵 Processing GTZAN files from {source_dir}")
    print(f"   Destination: {dest_dir}")
    print(f"   Cutting 30-second clips...")

    # Create destination directory
    Path(dest_dir).mkdir(parents=True, exist_ok=True)

    stats = {"total_files": 0, "successful": 0, "failed": 0, "genres": {}}

    # Walk through source directory
    for root, dirs, files in os.walk(source_dir):
        # Find WAV files
        wav_files = [f for f in files if f.lower().endswith(".wav")]

        if not wav_files:
            continue

        # Determine relative path from source directory
        rel_path = os.path.relpath(root, source_dir)
        genre_name = os.path.basename(root) if rel_path != "." else "root"

        # Skip the root directory itself if it's the source
        if root == source_dir:
            continue

        print(f"\nProcessing genre: {genre_name}")

        # Create corresponding directory in destination
        if maintain_structure and genre_name != "root":
            dest_genre_dir = os.path.join(dest_dir, genre_name)
            Path(dest_genre_dir).mkdir(parents=True, exist_ok=True)
            stats["genres"][genre_name] = {"total": 0, "successful": 0, "failed": 0}
        else:
            dest_genre_dir = dest_dir
            if genre_name not in stats["genres"]:
                stats["genres"][genre_name] = {"total": 0, "successful": 0, "failed": 0}

        # Process each WAV file
        for wav_file in tqdm(wav_files, desc=f"  {genre_name}"):
            source_file = os.path.join(root, wav_file)
            dest_file = os.path.join(dest_genre_dir, wav_file)

            stats["total_files"] += 1
            stats["genres"][genre_name]["total"] += 1

            # Cut and save the 30-second clip
            if cut_30_second_clip(source_file, dest_file):
                stats["successful"] += 1
                stats["genres"][genre_name]["successful"] += 1
            else:
                stats["failed"] += 1
                stats["genres"][genre_name]["failed"] += 1

    return stats


def extract_mfccs_from_processed(processed_dir: str, output_dir: str, output_filename: str) -> bool:
    """
    Extract MFCC features from processed GTZAN files.

    Args:
        processed_dir: Directory containing processed 30-second clips
        output_dir: Directory to save MFCC features
        output_filename: Name of the output file (without .json extension)

    Returns:
        True if successful, False otherwise
    """
    print(f"\n📊 Extracting MFCC features from {processed_dir}")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        # Process GTZAN dataset using existing function
        extracted_data = process_gtzan_dataset(
            music_path=processed_dir,
            mfcc_count=13,
            n_fft=2048,
            hop_length=512,
            seg_length=30,
        )

        # Save data to JSON file
        output_file = os.path.join(output_dir, f"{output_filename}.json")
        save_gtzan_data(extracted_data, output_file)

        print(f"\n✅ MFCC extraction completed successfully!")
        print(f"   Output saved to: {output_file}")

        return True

    except Exception as e:
        print(f"\n❌ Error during MFCC extraction: {e}")
        import traceback

        traceback.print_exc()
        return False


########################################################################
# MAIN FUNCTION
########################################################################


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Process GTZAN audio files: cut 30-second clips and extract MFCC features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full processing pipeline
  python process_gtzan_audio.py /home/eo/Documents/gtzan gtzan-data/processed gtzan-data/mfccs gtzan_features

  # Only cut clips (skip MFCC extraction)
  python process_gtzan_audio.py /home/eo/Documents/gtzan gtzan-data/processed --skip-mfcc

  # Only extract MFCCs from already processed files
  python process_gtzan_audio.py --skip-copy --mfcc-dir gtzan-data/processed --output-dir gtzan-data/mfccs --output-name gtzan_features
        """,
    )

    parser.add_argument(
        "source_dir", nargs="?", help="Path to the original GTZAN dataset directory"
    )
    parser.add_argument(
        "processed_dir",
        nargs="?",
        help="Path to save processed 30-second clips",
    )
    parser.add_argument(
        "output_dir", nargs="?", help="Path to save extracted MFCC features"
    )
    parser.add_argument("output_name", nargs="?", help="Output filename (without .json)")

    parser.add_argument(
        "--skip-copy",
        action="store_true",
        help="Skip the file copying/clipping step",
    )
    parser.add_argument(
        "--skip-mfcc",
        action="store_true",
        help="Skip the MFCC extraction step",
    )
    parser.add_argument(
        "--mfcc-dir",
        help="Directory containing processed files for MFCC extraction (if skipped copy)",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for MFCC features (if skipped copy)",
    )
    parser.add_argument(
        "--output-name",
        help="Output filename for MFCC features (if skipped copy)",
    )

    args = parser.parse_args()

    # Validation
    if not args.skip_copy:
        if not args.source_dir or not args.processed_dir:
            print("❌ Error: source_dir and processed_dir required when not skipping copy")
            return 1

        if not os.path.exists(args.source_dir):
            print(f"❌ Error: Source directory does not exist: {args.source_dir}")
            return 1

        # Create processed directory
        Path(args.processed_dir).mkdir(parents=True, exist_ok=True)

    # Step 1: Copy and cut files
    if not args.skip_copy:
        stats = copy_and_cut_files(args.source_dir, args.processed_dir)
        print(f"\n📊 File Processing Statistics:")
        print(f"   Total files: {stats['total_files']}")
        print(f"   Successful: {stats['successful']}")
        print(f"   Failed: {stats['failed']}")
        print(f"\n   Per-genre breakdown:")
        for genre, genre_stats in stats["genres"].items():
            print(
                f"   {genre}: {genre_stats['successful']}/{genre_stats['total']} files"
            )

    # Step 2: Extract MFCC features
    if not args.skip_mfcc:
        # Determine which directories to use
        if args.skip_copy:
            mfcc_source_dir = args.mfcc_dir or args.processed_dir
            mfcc_output_dir = args.output_dir
            mfcc_output_name = args.output_name or "gtzan_features"
        else:
            mfcc_source_dir = args.processed_dir
            mfcc_output_dir = args.output_dir
            mfcc_output_name = args.output_name or "gtzan_features"

        if not mfcc_source_dir:
            print("❌ Error: No source directory specified for MFCC extraction")
            return 1

        if not mfcc_output_dir:
            print("❌ Error: No output directory specified for MFCC extraction")
            return 1

        extract_mfccs_from_processed(mfcc_source_dir, mfcc_output_dir, mfcc_output_name)

    print("\n🎉 Processing completed successfully!")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

