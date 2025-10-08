#!/usr/bin/env python3
"""
FMA MFCC Extraction Script for Music Genre Classification

This script extracts MFCC features from the FMA dataset (MP3 files, JSON-based genre mapping).
The FMA dataset requires:
1. Track-to-genre mapping from FMA CSV
2. MP3-to-genre mapping creation
3. MFCC extraction with incremental saving
4. Label processing and validation

This script integrates functionality from:
- extract_fma_track_genres.py (track-to-genre mapping)
- map_mp3_to_genres.py (MP3-to-genre mapping)
- add_labels_to_fma.py (label processing)
- MFCC_extraction_unified.py (MFCC extraction)
"""

import sys
import os
import argparse

# Set thread limits BEFORE importing any libraries that might use threading
def set_early_thread_limits(max_threads: int = 8):
    """Set thread limits before importing heavy libraries."""
    print(f"🔧 Setting early thread limits to {max_threads} threads")
    
    # Set environment variables for different libraries
    os.environ['OMP_NUM_THREADS'] = str(max_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(max_threads)
    os.environ['MKL_NUM_THREADS'] = str(max_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(max_threads)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(max_threads)
    os.environ['NUMBA_NUM_THREADS'] = str(max_threads)
    os.environ['NUMBA_NUMBA_THREADING_LAYER'] = 'omp'
    
    # Try to set CPU affinity to limit cores (Linux only)
    try:
        total_cores = os.cpu_count()
        if total_cores and max_threads < total_cores:
            # Use only the first max_threads cores
            cpu_list = list(range(max_threads))
            os.sched_setaffinity(0, cpu_list)
            print(f"✅ CPU affinity set to cores: {cpu_list}")
    except (AttributeError, OSError) as e:
        print(f"⚠️  Warning: Could not set CPU affinity: {e}")
    
    print(f"✅ Early thread limits set successfully")

# Parse command line arguments early to get thread settings
def parse_thread_args():
    """Parse only thread-related arguments early."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--max-threads", type=int, default=8, help="Maximum number of threads to use")
    parser.add_argument("--no-thread-limit", action="store_true", help="Disable thread limiting")
    
    # Parse only known args to avoid errors with other arguments
    args, _ = parser.parse_known_args()
    return args

# Get thread settings and apply them immediately
thread_args = parse_thread_args()
if not thread_args.no_thread_limit:
    set_early_thread_limits(thread_args.max_threads)
else:
    print("🔧 Thread limiting disabled - using all available cores")

# Now import the heavy libraries after thread limits are set
import json
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import Counter
import threading

########################################################################
# CONSTANT VARIABLES
########################################################################

# Constants for audio processing
SAMPLE_RATE = 22050  # Standard sample rate for audio data
SONG_LENGTH = 30  # Duration of each song clip in seconds
SAMPLE_COUNT = SAMPLE_RATE * SONG_LENGTH  # Total number of samples per clip

########################################################################
# FMA DATA PROCESSING FUNCTIONS
########################################################################

def extract_track_genre_mapping(tracks_file: str, subset: str = "medium") -> Dict[str, str]:
    """
    Extract track-to-genre mapping from FMA tracks.csv file.
    
    Args:
        tracks_file: Path to tracks.csv file
        subset: Dataset subset to filter (small, medium, large)
        
    Returns:
        Dictionary mapping track IDs to genre names
    """
    print(f"📖 Extracting FMA track-to-genre mapping from {tracks_file}...")
    print(f"   Subset: {subset}")
    
    # Read the tracks.csv file with multi-level headers
    tracks_df = pd.read_csv(tracks_file, index_col=0, header=[0, 1])
    print(f"   Loaded {len(tracks_df)} tracks")
    
    # Filter by subset
    if subset == "small":
        subset_tracks = tracks_df[tracks_df[('set', 'subset')] <= 'small']
    elif subset == "medium":
        subset_tracks = tracks_df[tracks_df[('set', 'subset')] <= 'medium']
    else:
        subset_tracks = tracks_df[tracks_df[('set', 'subset')] == subset]
    
    print(f"   Found {len(subset_tracks)} tracks in {subset} subset")
    
    # Extract track-to-genre mapping
    track_genre_mapping = {}
    genre_counts = {}
    
    for track_id in subset_tracks.index:
        try:
            # Get the genre_top value
            genre = subset_tracks.loc[track_id, ('track', 'genre_top')]
            
            # Only include tracks with valid genres
            if pd.notna(genre) and genre != 'Unknown' and genre != '':
                track_genre_mapping[str(track_id)] = str(genre)
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
                
        except (KeyError, IndexError) as e:
            # Skip tracks without genre information
            continue
    
    print(f"   Extracted {len(track_genre_mapping)} tracks with valid genres")
    print(f"   Found {len(genre_counts)} unique genres")
    
    # Print genre distribution
    print("\n📊 Genre Distribution:")
    print("-" * 40)
    for genre, count in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {genre}: {count} tracks")
    
    return track_genre_mapping

def get_mp3_files(audio_dir: str) -> List[str]:
    """Get all MP3 files from the FMA directory structure."""
    mp3_files = []
    
    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if file.endswith('.mp3'):
                file_path = os.path.join(root, file)
                mp3_files.append(file_path)
    
    return mp3_files

def extract_track_id_from_filename(filename: str) -> Optional[int]:
    """Extract track ID from MP3 filename."""
    try:
        # Remove .mp3 extension and convert to int
        track_id = int(filename.replace('.mp3', ''))
        return track_id
    except ValueError:
        return None

def create_mp3_genre_mapping(audio_dir: str, track_genre_mapping: Dict[str, str]) -> Dict[str, str]:
    """
    Create mapping of MP3 files to their real FMA genres.
    
    Args:
        audio_dir: Path to FMA audio directory
        track_genre_mapping: Dictionary mapping track IDs to genres
        
    Returns:
        Dictionary mapping MP3 file paths to genre names
    """
    print("🔗 Creating MP3-to-genre mapping...")
    print(f"   Audio directory: {audio_dir}")
    
    # Get all MP3 files
    print("🔍 Scanning for MP3 files...")
    mp3_files = get_mp3_files(audio_dir)
    print(f"   Found {len(mp3_files)} MP3 files")
    
    # Create MP3-to-genre mapping
    mp3_genre_mapping = {}
    genre_counts = {}
    matched_files = 0
    
    for mp3_file in mp3_files:
        filename = os.path.basename(mp3_file)
        
        # Extract track ID from filename
        track_id = extract_track_id_from_filename(filename)
        
        if track_id is not None:
            track_id_str = str(track_id)
            
            # Get genre from track-genre mapping
            if track_id_str in track_genre_mapping:
                genre = track_genre_mapping[track_id_str]
                mp3_genre_mapping[mp3_file] = genre
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
                matched_files += 1
                
                if matched_files % 1000 == 0:
                    print(f"   Processed {matched_files} files...")
    
    print(f"   Matched {matched_files} MP3 files to genres")
    print(f"   Found {len(genre_counts)} unique genres")
    
    # Print genre distribution
    print("\n📊 MP3 Genre Distribution:")
    print("-" * 40)
    for genre, count in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {genre}: {count} files")
    
    return mp3_genre_mapping

def get_unique_genres(genre_mapping: Dict[str, str]) -> List[str]:
    """
    Get unique genres from genre mapping.
    
    Args:
        genre_mapping: Dictionary mapping files to genres
        
    Returns:
        Sorted list of unique genre names
    """
    unique_genres = list(set(genre_mapping.values()))
    return sorted(unique_genres)

def create_genre_to_index_mapping(genres: List[str]) -> Dict[str, int]:
    """
    Create mapping from genre names to numeric indices.
    
    Args:
        genres: List of unique genre names
        
    Returns:
        Dictionary mapping genre names to indices
    """
    return {genre: idx for idx, genre in enumerate(genres)}

########################################################################
# CORE MFCC EXTRACTION FUNCTIONS
########################################################################

def extract_mfcc_from_audio(
    audio_path: str,
    mfcc_count: int = 13,
    n_fft: int = 2048,
    hop_length: int = 512,
    seg_length: int = 30
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

def save_fma_data_incremental(
    extracted_data: Dict,
    output_file: str,
    processed_count: int,
    total_files: int
) -> None:
    """
    Save FMA data incrementally to the output file in GTZAN format (features and labels).
    
    Args:
        extracted_data: Current extracted data dictionary
        output_file: Path to output file
        processed_count: Number of files processed
        total_files: Total number of files to process
    """
    try:
        # Convert to GTZAN format (features and labels)
        gtzan_format_data = {
            "features": extracted_data["mfcc"],
            "labels": extracted_data["labels"]
        }
        
        with open(output_file, 'w') as f:
            json.dump(gtzan_format_data, f, indent=2)
        print(f"💾 Saved {processed_count}/{total_files} samples to {output_file}")
    except Exception as e:
        print(f"⚠️  Warning: Could not save data: {e}")

def load_existing_fma_data(output_file: str) -> Tuple[Dict, int, int, int]:
    """
    Load existing FMA data from the output file.
    
    Args:
        output_file: Path to the output JSON file
        
    Returns:
        Tuple of (extracted_data, processed_count, skipped_count, total_files)
    """
    if not os.path.exists(output_file):
        return None, 0, 0, 0
    
    try:
        with open(output_file, 'r') as f:
            file_data = json.load(f)
        
        # Check if it's the new GTZAN format (features only) or old format
        if "features" in file_data:
            # New GTZAN format - convert back to internal format
            processed_count = len(file_data["features"])
            print(f"📂 Found existing FMA data (GTZAN format): {processed_count} samples already processed")
            
            # We can't resume from GTZAN format as we don't have the internal structure
            # Return None to start fresh
            return None, 0, 0, 0
        else:
            # Old format - load normally
            extracted_data = file_data
            processed_count = len(extracted_data.get("mfcc", []))
            skipped_count = 0  # We don't track skipped files in the final output
            
            print(f"📂 Found existing FMA data (old format): {processed_count} samples already processed")
            print(f"   Genres: {extracted_data.get('mapping', [])}")
            
            return extracted_data, processed_count, skipped_count, 0  # total_files unknown from existing data
    except Exception as e:
        print(f"⚠️  Warning: Could not load existing data: {e}")
        return None, 0, 0, 0

def process_fma_dataset(
    music_path: str,
    tracks_file: str,
    output_file: str,
    mfcc_count: int = 13,
    n_fft: int = 2048,
    hop_length: int = 512,
    seg_length: int = 30,
    checkpoint_interval: int = 1000,
    subset: str = "medium"
) -> Dict:
    """
    Process FMA dataset (MP3 files, JSON-based genre mapping) with incremental saving.
    
    Args:
        music_path: Path to FMA dataset directory
        tracks_file: Path to FMA tracks.csv file
        output_file: Path to output JSON file
        mfcc_count: Number of MFCC coefficients
        n_fft: FFT window size
        hop_length: Number of samples between successive frames
        seg_length: Length of audio segment in seconds
        checkpoint_interval: Save checkpoint every N files
        subset: FMA dataset subset (small, medium, large)
        
    Returns:
        Dictionary containing extracted data
    """
    print("🎵 Processing FMA dataset (MP3 files, JSON-based genres)...")
    
    # Step 1: Extract track-to-genre mapping from CSV
    track_genre_mapping = extract_track_genre_mapping(tracks_file, subset)
    
    # Step 2: Create MP3-to-genre mapping
    mp3_genre_mapping = create_mp3_genre_mapping(music_path, track_genre_mapping)
    
    # Step 3: Get unique genres and create index mapping
    unique_genres = get_unique_genres(mp3_genre_mapping)
    genre_to_index = create_genre_to_index_mapping(unique_genres)
    
    print(f"   Loaded {len(mp3_genre_mapping)} MP3-genre mappings")
    print(f"   Found {len(unique_genres)} unique genres: {unique_genres}")
    
    # Try to load existing data
    extracted_data, processed_count, skipped_count, _ = load_existing_fma_data(output_file)
    
    if extracted_data is None:
        # Initialize the data dictionary
        extracted_data = {
            "dataset_type": "fma",
            "mapping": unique_genres,  # List to map numeric labels to genre names
            "labels": [],              # List to store numeric labels for each audio clip
            "mfcc": [],                # List to store extracted MFCCs
        }
        processed_count = 0
        skipped_count = 0
        total_files = len(mp3_genre_mapping)
        print(f"🆕 Starting fresh extraction of {total_files} files")
    else:
        # Determine total files from mapping
        total_files = len(mp3_genre_mapping)
        print(f"🔄 Resuming from existing data: {processed_count}/{total_files} files already processed")
    
    # Convert mp3_genre_mapping to list for indexing
    mp3_files_list = list(mp3_genre_mapping.items())
    
    # Process remaining files
    for i, (mp3_file, genre) in enumerate(mp3_files_list[processed_count + skipped_count:], 
                                         start=processed_count + skipped_count):
        # Check if the file exists
        if not os.path.exists(mp3_file):
            print(f"Warning: File not found: {mp3_file}")
            skipped_count += 1
            continue
        
        # Extract MFCC features
        mfcc = extract_mfcc_from_audio(
            mp3_file, mfcc_count, n_fft, hop_length, seg_length
        )
        
        if mfcc is not None:
            # Get genre index
            genre_index = genre_to_index[genre]
            
            # Append MFCCs and label to the data dictionary
            extracted_data["mfcc"].append(mfcc.tolist())
            extracted_data["labels"].append(genre_index)
            processed_count += 1
            
            if processed_count % 100 == 0:
                print(f"  Processed {processed_count} files...")
        else:
            skipped_count += 1
        
        # Save incrementally every checkpoint_interval files
        if (processed_count + skipped_count) % checkpoint_interval == 0:
            save_fma_data_incremental(extracted_data, output_file, processed_count, total_files)
    
    # Save final data
    save_fma_data_incremental(extracted_data, output_file, processed_count, total_files)

    print(f"\n✅ FMA processing complete!")
    print(f"   Genres: {unique_genres}")
    print(f"   Processed samples: {processed_count}")
    print(f"   Skipped files: {skipped_count}")
    print(f"   Output file: {output_file}")
    
    return extracted_data

########################################################################
# MAIN FUNCTION
########################################################################

def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="FMA MFCC extraction for music genre classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process FMA medium subset with default parameters (8 threads)
  python MFCC_FMA_extract.py /path/to/fma /path/to/tracks.csv /path/to/output fma_features

  # Process FMA large subset with custom parameters
  python MFCC_FMA_extract.py /path/to/fma /path/to/tracks.csv /path/to/output fma_features --subset large --mfcc-count 20

  # Process with custom segment length and checkpoint interval
  python MFCC_FMA_extract.py /path/to/fma /path/to/tracks.csv /path/to/output fma_features --seg-length 45 --checkpoint-interval 500

  # Process with custom thread count (4 threads for lower CPU usage)
  python MFCC_FMA_extract.py /path/to/fma /path/to/tracks.csv /path/to/output fma_features --max-threads 4

  # Process with no thread limiting (use all available cores)
  python MFCC_FMA_extract.py /path/to/fma /path/to/tracks.csv /path/to/output fma_features --no-thread-limit
        """
    )
    
    parser.add_argument("music_path", help="Path to the FMA dataset directory")
    parser.add_argument("tracks_file", help="Path to FMA tracks.csv file")
    parser.add_argument("output_path", help="Path to save the output JSON file")
    parser.add_argument("output_filename", help="Name of the output file (without .json extension)")
    
    parser.add_argument("--subset", default="medium", choices=["small", "medium", "large"], 
                       help="FMA dataset subset (default: medium)")
    parser.add_argument("--mfcc-count", type=int, default=13, help="Number of MFCC coefficients (default: 13)")
    parser.add_argument("--n-fft", type=int, default=2048, help="FFT window size (default: 2048)")
    parser.add_argument("--hop-length", type=int, default=512, help="Hop length (default: 512)")
    parser.add_argument("--seg-length", type=int, default=30, help="Segment length in seconds (default: 30)")
    parser.add_argument("--checkpoint-interval", type=int, default=1000, help="Save checkpoint every N files (default: 1000)")
    parser.add_argument("--max-threads", type=int, default=8, help="Maximum number of threads to use (default: 8)")
    parser.add_argument("--no-thread-limit", action="store_true", help="Disable thread limiting (use all available cores)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.music_path):
        print(f"❌ Error: Music path does not exist: {args.music_path}")
        return 1
    
    if not os.path.exists(args.tracks_file):
        print(f"❌ Error: Tracks file does not exist: {args.tracks_file}")
        return 1
    
    if not os.path.exists(args.output_path):
        print(f"❌ Error: Output path does not exist: {args.output_path}")
        return 1
    
    try:
        # Process FMA dataset
        output_file = os.path.join(args.output_path, args.output_filename + ".json")
        extracted_data = process_fma_dataset(
            music_path=args.music_path,
            tracks_file=args.tracks_file,
            output_file=output_file,
            mfcc_count=args.mfcc_count,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            seg_length=args.seg_length,
            checkpoint_interval=args.checkpoint_interval,
            subset=args.subset
        )
        
        print("\n🎉 FMA MFCC extraction completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Error during MFCC extraction: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
