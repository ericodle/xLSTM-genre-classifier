#!/usr/bin/env python3
"""
Map MP3 filenames to genres using the FMA track-genre mapping.
This creates a mapping of actual MP3 files in fma_medium/ to their real FMA genres.
"""

import json
import os
import sys
from pathlib import Path

def load_track_genre_mapping(json_file: str):
    """Load the track-to-genre mapping from JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data['track_genre_mapping'], data['metadata']

def get_mp3_files(audio_dir: str):
    """Get all MP3 files from the fma_medium directory structure."""
    mp3_files = []
    
    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if file.endswith('.mp3'):
                file_path = os.path.join(root, file)
                mp3_files.append(file_path)
    
    return mp3_files

def extract_track_id_from_filename(filename: str):
    """Extract track ID from MP3 filename."""
    try:
        # Remove .mp3 extension and convert to int
        track_id = int(filename.replace('.mp3', ''))
        return track_id
    except ValueError:
        return None

def create_mp3_genre_mapping(audio_dir: str, track_genre_json: str, output_file: str):
    """
    Create mapping of MP3 files to their real FMA genres.
    
    Args:
        audio_dir: Path to fma_medium directory
        track_genre_json: Path to track-genre mapping JSON file
        output_file: Path to output MP3-genre mapping JSON file
    """
    
    print("🎵 Creating MP3-to-genre mapping...")
    print(f"   Audio directory: {audio_dir}")
    print(f"   Track-genre mapping: {track_genre_json}")
    print(f"   Output file: {output_file}")
    
    # Load track-to-genre mapping
    print("📖 Loading track-to-genre mapping...")
    track_genre_mapping, metadata = load_track_genre_mapping(track_genre_json)
    
    print(f"   Loaded {len(track_genre_mapping)} track-genre mappings")
    print(f"   Genres: {metadata['unique_genres']}")
    
    # Get all MP3 files
    print("🔍 Scanning for MP3 files...")
    mp3_files = get_mp3_files(audio_dir)
    print(f"   Found {len(mp3_files)} MP3 files")
    
    # Create MP3-to-genre mapping
    print("🔗 Creating MP3-to-genre mapping...")
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
    
    # Create output data
    output_data = {
        "metadata": {
            "source": "FMA MP3 files mapped to real genres",
            "audio_directory": audio_dir,
            "total_mp3_files": len(mp3_files),
            "matched_files": matched_files,
            "unique_genres": len(genre_counts),
            "genres": list(genre_counts.keys())
        },
        "mp3_genre_mapping": mp3_genre_mapping,
        "genre_distribution": genre_counts
    }
    
    # Save to JSON file
    print(f"💾 Saving MP3-to-genre mapping to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print genre distribution
    print("\n📊 Genre Distribution (MP3 files):")
    print("-" * 40)
    for genre, count in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {genre}: {count} files")
    
    print(f"\n✅ Successfully created MP3-to-genre mapping!")
    print(f"   Output file: {output_file}")
    print(f"   Total MP3 files: {len(mp3_files)}")
    print(f"   Matched files: {matched_files}")
    print(f"   Unique genres: {len(genre_counts)}")
    
    return output_data

def verify_mp3_genre_mapping(json_file: str, num_samples: int = 20):
    """
    Verify the MP3-to-genre mapping by showing sample files.
    """
    
    print("🎵 Verifying MP3-to-genre mapping...")
    
    # Load the mapping
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    mp3_genre_mapping = data['mp3_genre_mapping']
    metadata = data['metadata']
    
    print(f"   Total MP3 files: {metadata['total_mp3_files']}")
    print(f"   Matched files: {metadata['matched_files']}")
    print(f"   Unique genres: {metadata['unique_genres']}")
    
    # Show sample files
    print(f"\n📋 Sample MP3 files with genres:")
    print("-" * 50)
    
    sample_files = list(mp3_genre_mapping.items())[:num_samples]
    
    for i, (mp3_file, genre) in enumerate(sample_files, 1):
        filename = os.path.basename(mp3_file)
        exists = "✅" if os.path.exists(mp3_file) else "❌"
        
        print(f"{i:2d}. {exists} {filename}")
        print(f"    Genre: {genre}")
        print(f"    Path: {mp3_file}")
        print()

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Map MP3 files to FMA genres")
    parser.add_argument("--audio-dir", default="mfccs/fma_medium", help="Path to fma_medium directory")
    parser.add_argument("--track-genres", default="mfccs/fma_track_genres.json", help="Path to track-genre mapping JSON file")
    parser.add_argument("--output", default="mfccs/fma_mp3_genres.json", help="Output MP3-genre mapping JSON file")
    parser.add_argument("--verify", action="store_true", help="Verify the mapping by showing samples")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples to show during verification")
    
    args = parser.parse_args()
    
    # Check if input files exist
    if not os.path.exists(args.audio_dir):
        print(f"❌ Error: Audio directory not found: {args.audio_dir}")
        return 1
    
    if not os.path.exists(args.track_genres):
        print(f"❌ Error: Track-genre mapping file not found: {args.track_genres}")
        return 1
    
    try:
        if args.verify:
            # Verify existing mapping
            if os.path.exists(args.output):
                verify_mp3_genre_mapping(args.output, args.samples)
            else:
                print(f"❌ Error: Output file not found: {args.output}")
                print("Run without --verify to create the mapping first")
                return 1
        else:
            # Create mapping
            result = create_mp3_genre_mapping(
                audio_dir=args.audio_dir,
                track_genre_json=args.track_genres,
                output_file=args.output
            )
            
            # Auto-verify
            print("\n" + "="*60)
            verify_mp3_genre_mapping(args.output, args.samples)
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
