#!/usr/bin/env python3
"""
Unified MFCC Extraction Script for Music Genre Classification

This script can extract MFCC features from both:
1. GTZAN dataset (WAV files, folder-based genre labels)
2. FMA dataset (MP3 files, JSON-based genre mapping)

The script automatically detects the dataset type and processes accordingly.
"""

import sys
import json
import os
import librosa
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

########################################################################
# CONSTANT VARIABLES
########################################################################

# Constants for audio processing
SAMPLE_RATE = 22050  # Standard sample rate for audio data
SONG_LENGTH = 30  # Duration of each song clip in seconds
SAMPLE_COUNT = SAMPLE_RATE * SONG_LENGTH  # Total number of samples per clip

# Dataset type constants
DATASET_GTZAN = "gtzan"
DATASET_FMA = "fma"

########################################################################
# UTILITY FUNCTIONS
########################################################################

def detect_dataset_type(music_path: str) -> str:
    """
    Detect dataset type based on directory structure and file types.
    
    Args:
        music_path: Path to the music dataset directory
        
    Returns:
        Dataset type ('gtzan' or 'fma')
    """
    if not os.path.exists(music_path):
        raise ValueError(f"Music path does not exist: {music_path}")
    
    # Check for FMA structure (has subdirectories with numbers, contains MP3 files)
    mp3_count = 0
    wav_count = 0
    subdir_count = 0
    
    for root, dirs, files in os.walk(music_path):
        # Count subdirectories (FMA has numbered subdirs like 000, 001, etc.)
        if root != music_path:
            subdir_count += 1
        
        # Count file types
        for file in files:
            if file.endswith('.mp3'):
                mp3_count += 1
            elif file.endswith('.wav'):
                wav_count += 1
    
    # FMA typically has many subdirectories and MP3 files
    if mp3_count > 0 and subdir_count > 10:
        return DATASET_FMA
    # GTZAN has genre folders and WAV files
    elif wav_count > 0 and subdir_count <= 10:
        return DATASET_GTZAN
    else:
        # Default to GTZAN if unclear
        print(f"Warning: Could not clearly detect dataset type. Found {mp3_count} MP3s, {wav_count} WAVs, {subdir_count} subdirs")
        print("Defaulting to GTZAN format...")
        return DATASET_GTZAN

def load_fma_genre_mapping(genre_mapping_file: str) -> Dict[str, str]:
    """
    Load FMA MP3-to-genre mapping from JSON file.
    
    Args:
        genre_mapping_file: Path to the FMA genre mapping JSON file
        
    Returns:
        Dictionary mapping MP3 file paths to genre names
    """
    if not os.path.exists(genre_mapping_file):
        raise FileNotFoundError(f"FMA genre mapping file not found: {genre_mapping_file}")
    
    with open(genre_mapping_file, 'r') as f:
        data = json.load(f)
    
    return data['mp3_genre_mapping']

def get_unique_genres_fma(genre_mapping: Dict[str, str]) -> List[str]:
    """
    Get unique genres from FMA genre mapping.
    
    Args:
        genre_mapping: Dictionary mapping MP3 files to genres
        
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

def process_gtzan_dataset(
    music_path: str,
    mfcc_count: int = 13,
    n_fft: int = 2048,
    hop_length: int = 512,
    seg_length: int = 30
) -> Dict:
    """
    Process GTZAN dataset (WAV files, folder-based genres).
    
    Args:
        music_path: Path to GTZAN dataset directory
        mfcc_count: Number of MFCC coefficients
        n_fft: FFT window size
        hop_length: Number of samples between successive frames
        seg_length: Length of audio segment in seconds
        
    Returns:
        Dictionary containing extracted data
    """
    print("🎵 Processing GTZAN dataset (WAV files, folder-based genres)...")
    
    # Initialize the data dictionary
    extracted_data = {
        "dataset_type": "gtzan",
        "mapping": [],  # List to map numeric labels to genre names
        "labels": [],   # List to store numeric labels for each audio clip
        "mfcc": [],     # List to store extracted MFCCs
    }

    # Process each genre folder
    for i, (folder_path, folder_name, file_name) in enumerate(os.walk(music_path)):
        if folder_path != music_path:
            # Extract genre label from folder path
            genre_label = folder_path.split("/")[-1]
            extracted_data["mapping"].append(genre_label)
            print(f"\nProcessing: {genre_label}")

            # Iterate over each audio file in the genre folder
            for song_clip in file_name:
                if song_clip.endswith('.wav'):
                    file_path = os.path.join(folder_path, song_clip)
                    
                    # Extract MFCC features
                    mfcc = extract_mfcc_from_audio(
                        file_path, mfcc_count, n_fft, hop_length, seg_length
                    )
                    
                    if mfcc is not None:
                        # Append MFCCs and label to the data dictionary
                        extracted_data["mfcc"].append(mfcc.tolist())
                        extracted_data["labels"].append(i - 1)
                        print(f"  {song_clip}")

    print(f"\n✅ GTZAN processing complete!")
    print(f"   Genres: {extracted_data['mapping']}")
    print(f"   Total samples: {len(extracted_data['mfcc'])}")
    
    return extracted_data

def save_checkpoint(
    extracted_data: Dict,
    checkpoint_file: str,
    processed_count: int,
    skipped_count: int,
    total_files: int
) -> None:
    """
    Save checkpoint data to file.
    
    Args:
        extracted_data: Current extracted data dictionary
        checkpoint_file: Path to checkpoint file
        processed_count: Number of files processed
        skipped_count: Number of files skipped
        total_files: Total number of files to process
    """
    checkpoint_data = {
        "extracted_data": extracted_data,
        "progress": {
            "processed_count": processed_count,
            "skipped_count": skipped_count,
            "total_files": total_files,
            "remaining_files": total_files - processed_count - skipped_count
        },
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }
    
    try:
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        print(f"💾 Checkpoint saved: {processed_count}/{total_files} files processed")
    except Exception as e:
        print(f"⚠️  Warning: Could not save checkpoint: {e}")

def load_checkpoint(checkpoint_file: str) -> Tuple[Dict, int, int, int]:
    """
    Load checkpoint data from file.
    
    Args:
        checkpoint_file: Path to checkpoint file
        
    Returns:
        Tuple of (extracted_data, processed_count, skipped_count, total_files)
    """
    if not os.path.exists(checkpoint_file):
        return None, 0, 0, 0
    
    try:
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
        
        extracted_data = checkpoint_data["extracted_data"]
        progress = checkpoint_data["progress"]
        
        print(f"📂 Checkpoint loaded: {progress['processed_count']}/{progress['total_files']} files processed")
        print(f"   Timestamp: {checkpoint_data.get('timestamp', 'unknown')}")
        
        return (
            extracted_data,
            progress["processed_count"],
            progress["skipped_count"],
            progress["total_files"]
        )
    except Exception as e:
        print(f"⚠️  Warning: Could not load checkpoint: {e}")
        return None, 0, 0, 0

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

def process_fma_dataset(
    music_path: str,
    genre_mapping_file: str,
    output_file: str,
    mfcc_count: int = 13,
    n_fft: int = 2048,
    hop_length: int = 512,
    seg_length: int = 30,
    checkpoint_interval: int = 1000
) -> Dict:
    """
    Process FMA dataset (MP3 files, JSON-based genre mapping) with incremental saving.
    
    Args:
        music_path: Path to FMA dataset directory
        genre_mapping_file: Path to FMA genre mapping JSON file
        output_file: Path to output JSON file
        mfcc_count: Number of MFCC coefficients
        n_fft: FFT window size
        hop_length: Number of samples between successive frames
        seg_length: Length of audio segment in seconds
        checkpoint_interval: Save checkpoint every N files
        
    Returns:
        Dictionary containing extracted data
    """
    print("🎵 Processing FMA dataset (MP3 files, JSON-based genres)...")
    
    # Load genre mapping
    print("📖 Loading FMA genre mapping...")
    mp3_genre_mapping = load_fma_genre_mapping(genre_mapping_file)
    unique_genres = get_unique_genres_fma(mp3_genre_mapping)
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

def mfcc_to_json_unified(
    music_path: str,
    output_path: str,
    output_filename: str,
    genre_mapping_file: Optional[str] = None,
    mfcc_count: int = 13,
    n_fft: int = 2048,
    hop_length: int = 512,
    seg_length: int = 30,
    dataset_type: Optional[str] = None,
    checkpoint_interval: int = 1000
) -> None:
    """
    Unified MFCC extraction function that handles both GTZAN and FMA datasets.
    
    Args:
        music_path: Path to the music dataset directory
        output_path: Path to save the output JSON file
        output_filename: Name of the output file (without extension)
        genre_mapping_file: Path to FMA genre mapping JSON file (required for FMA)
        mfcc_count: Number of MFCC coefficients
        n_fft: FFT window size
        hop_length: Number of samples between successive frames
        seg_length: Length of audio segment in seconds
        dataset_type: Force dataset type ('gtzan' or 'fma'), auto-detect if None
    """
    
    # Auto-detect dataset type if not specified
    if dataset_type is None:
        dataset_type = detect_dataset_type(music_path)
    
    print(f"🔍 Detected dataset type: {dataset_type.upper()}")
    
    # Process based on dataset type
    if dataset_type == DATASET_GTZAN:
        extracted_data = process_gtzan_dataset(
            music_path, mfcc_count, n_fft, hop_length, seg_length
        )
    elif dataset_type == DATASET_FMA:
        if genre_mapping_file is None:
            raise ValueError("FMA dataset requires genre_mapping_file parameter")
        output_file = os.path.join(output_path, output_filename + ".json")
        extracted_data = process_fma_dataset(
            music_path, genre_mapping_file, output_file, mfcc_count, n_fft, hop_length, seg_length, checkpoint_interval
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Write the extracted data to a JSON file (only for GTZAN, FMA is saved incrementally)
    if dataset_type == DATASET_GTZAN:
        output_filename = output_filename + ".json"
        output_file_path = os.path.join(output_path, output_filename)
        
        try:
            with open(output_file_path, "w") as fp:
                json.dump(extracted_data, fp, indent=4)
                print(f"\n💾 Successfully wrote data to {output_file_path}")
        except Exception as e:
            print(f"❌ Error writing data to {output_file_path}: {e}")

########################################################################
# MAIN FUNCTION
########################################################################

def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Unified MFCC extraction for GTZAN and FMA datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process GTZAN dataset (auto-detect)
  python MFCC_extraction_unified.py /path/to/gtzan /path/to/output gtzan_features

  # Process FMA dataset (auto-detect)
  python MFCC_extraction_unified.py /path/to/fma /path/to/output fma_features --genre-mapping /path/to/fma_mp3_genres.json

  # Force dataset type
  python MFCC_extraction_unified.py /path/to/data /path/to/output features --dataset-type fma --genre-mapping /path/to/mapping.json

  # Custom MFCC parameters
  python MFCC_extraction_unified.py /path/to/data /path/to/output features --mfcc-count 20 --n-fft 4096
        """
    )
    
    parser.add_argument("music_path", help="Path to the music dataset directory")
    parser.add_argument("output_path", help="Path to save the output JSON file")
    parser.add_argument("output_filename", help="Name of the output file (without .json extension)")
    
    parser.add_argument("--genre-mapping", help="Path to FMA genre mapping JSON file (required for FMA dataset)")
    parser.add_argument("--dataset-type", choices=[DATASET_GTZAN, DATASET_FMA], 
                       help="Force dataset type (auto-detect if not specified)")
    
    parser.add_argument("--mfcc-count", type=int, default=13, help="Number of MFCC coefficients (default: 13)")
    parser.add_argument("--n-fft", type=int, default=2048, help="FFT window size (default: 2048)")
    parser.add_argument("--hop-length", type=int, default=512, help="Hop length (default: 512)")
    parser.add_argument("--seg-length", type=int, default=30, help="Segment length in seconds (default: 30)")
    parser.add_argument("--checkpoint-interval", type=int, default=1000, help="Save checkpoint every N files (default: 1000)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.music_path):
        print(f"❌ Error: Music path does not exist: {args.music_path}")
        return 1
    
    if not os.path.exists(args.output_path):
        print(f"❌ Error: Output path does not exist: {args.output_path}")
        return 1
    
    # Check if FMA dataset type requires genre mapping
    if args.dataset_type == DATASET_FMA and args.genre_mapping is None:
        print("❌ Error: FMA dataset type requires --genre-mapping parameter")
        return 1
    
    try:
        # Call the unified extraction function
        mfcc_to_json_unified(
            music_path=args.music_path,
            output_path=args.output_path,
            output_filename=args.output_filename,
            genre_mapping_file=args.genre_mapping,
            mfcc_count=args.mfcc_count,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            seg_length=args.seg_length,
            dataset_type=args.dataset_type,
            checkpoint_interval=args.checkpoint_interval
        )
        
        print("\n🎉 MFCC extraction completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Error during MFCC extraction: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
