#!/usr/bin/env python3
"""
GTZAN MFCC Extraction Script for Music Genre Classification

This script extracts MFCC features from the GTZAN dataset (WAV files, folder-based genre labels).
The GTZAN dataset has a simple structure with genre folders containing WAV files.
"""

import sys
import json
import os
import librosa
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

########################################################################
# CONSTANT VARIABLES
########################################################################

# Constants for audio processing
SAMPLE_RATE = 22050  # Standard sample rate for audio data
SONG_LENGTH = 30  # Duration of each song clip in seconds
SAMPLE_COUNT = SAMPLE_RATE * SONG_LENGTH  # Total number of samples per clip

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
    
    # First pass: collect all genre folders and create mapping
    genre_folders = []
    for root, dirs, files in os.walk(music_path):
        if root != music_path and any(f.endswith('.wav') for f in files):
            genre_label = os.path.basename(root)
            genre_folders.append((root, genre_label))
    
    # Sort genres for consistent ordering
    genre_folders.sort(key=lambda x: x[1])
    
    # Create genre-to-index mapping
    genre_to_index = {genre: idx for idx, (_, genre) in enumerate(genre_folders)}
    unique_genres = [genre for _, genre in genre_folders]
    
    print(f"   Found {len(unique_genres)} genres: {unique_genres}")
    
    # Initialize the data dictionary
    extracted_data = {
        "dataset_type": "gtzan",
        "mapping": unique_genres,  # List to map numeric labels to genre names
        "labels": [],              # List to store numeric labels for each audio clip
        "mfcc": [],                # List to store extracted MFCCs
    }

    # Process each genre folder
    for folder_path, genre_label in genre_folders:
        genre_index = genre_to_index[genre_label]
        print(f"\nProcessing: {genre_label} (index: {genre_index})")

        # Get all WAV files in this genre folder
        wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        
        # Iterate over each audio file in the genre folder
        for song_clip in wav_files:
            file_path = os.path.join(folder_path, song_clip)
            
            # Extract MFCC features
            mfcc = extract_mfcc_from_audio(
                file_path, mfcc_count, n_fft, hop_length, seg_length
            )
            
            if mfcc is not None:
                # Append MFCCs and label to the data dictionary
                extracted_data["mfcc"].append(mfcc.tolist())
                extracted_data["labels"].append(genre_index)  # Use proper integer index
                print(f"  {song_clip}")

    print(f"\n✅ GTZAN processing complete!")
    print(f"   Genres: {extracted_data['mapping']}")
    print(f"   Total samples: {len(extracted_data['mfcc'])}")
    
    return extracted_data

def save_gtzan_data(extracted_data: Dict, output_file: str) -> None:
    """
    Save GTZAN data to JSON file in the same format as FMA (features and labels arrays).
    
    Args:
        extracted_data: Extracted data dictionary
        output_file: Path to output JSON file
    """
    try:
        # Convert to FMA format (features and labels arrays)
        gtzan_format_data = {
            "features": extracted_data["mfcc"],
            "labels": extracted_data["labels"],
            "mapping": extracted_data["mapping"]  # Include genre names
        }
        
        with open(output_file, "w") as fp:
            json.dump(gtzan_format_data, fp, indent=2)
            print(f"\n💾 Successfully wrote data to {output_file}")
    except Exception as e:
        print(f"❌ Error writing data to {output_file}: {e}")

########################################################################
# MAIN FUNCTION
########################################################################

def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="GTZAN MFCC extraction for music genre classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process GTZAN dataset with default parameters
  python MFCC_GTZAN_extract.py /path/to/gtzan /path/to/output gtzan_features

  # Process with custom MFCC parameters
  python MFCC_GTZAN_extract.py /path/to/gtzan /path/to/output gtzan_features --mfcc-count 20 --n-fft 4096

  # Process with custom segment length
  python MFCC_GTZAN_extract.py /path/to/gtzan /path/to/output gtzan_features --seg-length 45
        """
    )
    
    parser.add_argument("music_path", help="Path to the GTZAN dataset directory")
    parser.add_argument("output_path", help="Path to save the output JSON file")
    parser.add_argument("output_filename", help="Name of the output file (without .json extension)")
    
    parser.add_argument("--mfcc-count", type=int, default=13, help="Number of MFCC coefficients (default: 13)")
    parser.add_argument("--n-fft", type=int, default=2048, help="FFT window size (default: 2048)")
    parser.add_argument("--hop-length", type=int, default=512, help="Hop length (default: 512)")
    parser.add_argument("--seg-length", type=int, default=30, help="Segment length in seconds (default: 30)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.music_path):
        print(f"❌ Error: Music path does not exist: {args.music_path}")
        return 1
    
    if not os.path.exists(args.output_path):
        print(f"❌ Error: Output path does not exist: {args.output_path}")
        return 1
    
    try:
        # Process GTZAN dataset
        extracted_data = process_gtzan_dataset(
            music_path=args.music_path,
            mfcc_count=args.mfcc_count,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            seg_length=args.seg_length
        )
        
        # Save data to JSON file
        output_filename = args.output_filename + ".json"
        output_file_path = os.path.join(args.output_path, output_filename)
        save_gtzan_data(extracted_data, output_file_path)
        
        print("\n🎉 GTZAN MFCC extraction completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Error during MFCC extraction: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
