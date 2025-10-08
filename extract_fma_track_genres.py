#!/usr/bin/env python3
"""
Extract track-to-genre mapping from FMA metadata files.
This script reads the FMA tracks.csv file and creates a JSON mapping of track IDs to genre labels.
"""

import pandas as pd
import json
import os
import sys
from pathlib import Path

def extract_track_genre_mapping(tracks_file: str, output_file: str, subset: str = "medium"):
    """
    Extract track-to-genre mapping from FMA tracks.csv file.
    
    Args:
        tracks_file: Path to tracks.csv file
        output_file: Path to output JSON file
        subset: Dataset subset to filter (small, medium, large)
    """
    
    print(f"🎵 Extracting FMA track-to-genre mapping...")
    print(f"   Input: {tracks_file}")
    print(f"   Output: {output_file}")
    print(f"   Subset: {subset}")
    
    # Read the tracks.csv file with multi-level headers
    print("📖 Reading tracks.csv file...")
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
    print("🔍 Extracting track-to-genre mapping...")
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
    
    # Create output data
    output_data = {
        "metadata": {
            "source": "FMA tracks.csv",
            "subset": subset,
            "total_tracks": len(track_genre_mapping),
            "unique_genres": len(genre_counts),
            "genres": list(genre_counts.keys())
        },
        "track_genre_mapping": track_genre_mapping,
        "genre_distribution": genre_counts
    }
    
    # Save to JSON file
    print(f"💾 Saving mapping to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print genre distribution
    print("\n📊 Genre Distribution:")
    print("-" * 40)
    for genre, count in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {genre}: {count} tracks")
    
    print(f"\n✅ Successfully created track-to-genre mapping!")
    print(f"   Output file: {output_file}")
    print(f"   Total tracks: {len(track_genre_mapping)}")
    print(f"   Unique genres: {len(genre_counts)}")
    
    return output_data

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract FMA track-to-genre mapping")
    parser.add_argument("--tracks-file", default="mfccs/tracks.csv", help="Path to tracks.csv file")
    parser.add_argument("--output", default="mfccs/fma_track_genres.json", help="Output JSON file")
    parser.add_argument("--subset", default="medium", choices=["small", "medium", "large"], help="Dataset subset")
    
    args = parser.parse_args()
    
    # Check if tracks file exists
    if not os.path.exists(args.tracks_file):
        print(f"❌ Error: Tracks file not found: {args.tracks_file}")
        return 1
    
    try:
        result = extract_track_genre_mapping(
            tracks_file=args.tracks_file,
            output_file=args.output,
            subset=args.subset
        )
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
