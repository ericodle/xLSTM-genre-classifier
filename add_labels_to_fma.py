#!/usr/bin/env python3
"""
Add labels to existing FMA MFCC data using the genre mapping.
"""

import json
import os
from collections import Counter

def add_labels_to_fma_data():
    """Add labels to existing FMA MFCC data."""
    
    # Load the existing FMA MFCC data
    print("📖 Loading existing FMA MFCC data...")
    with open('mfccs/fma_13.json', 'r') as f:
        fma_data = json.load(f)
    
    print(f"   Found {len(fma_data['features'])} features")
    
    # Load the genre mapping
    print("📖 Loading FMA genre mapping...")
    with open('mfccs/fma_mp3_genres.json', 'r') as f:
        genre_mapping_data = json.load(f)
    
    mp3_genre_mapping = genre_mapping_data['mp3_genre_mapping']
    genres = genre_mapping_data['metadata']['genres']
    
    print(f"   Loaded {len(mp3_genre_mapping)} MP3-genre mappings")
    print(f"   Found {len(genres)} unique genres: {genres}")
    
    # Create genre to index mapping
    genre_to_index = {genre: idx for idx, genre in enumerate(genres)}
    print(f"   Genre mapping: {genre_to_index}")
    
    # Create index to filename mapping from the features
    # We need to match the features to the MP3 files
    # The features are stored in order, so we need to find the corresponding MP3 files
    
    # Get all MP3 files in the fma_medium directory
    fma_medium_path = "mfccs/fma_medium"
    mp3_files = []
    for root, dirs, files in os.walk(fma_medium_path):
        for file in files:
            if file.endswith('.mp3'):
                mp3_files.append(os.path.join(root, file))
    
    mp3_files.sort()
    print(f"   Found {len(mp3_files)} MP3 files in directory")
    
    # Match features to MP3 files and create labels
    labels = []
    matched_count = 0
    skipped_count = 0
    
    print("🔄 Matching features to genres...")
    
    # Create a mapping from filename to genre for faster lookup
    filename_to_genre = {}
    for mp3_path, genre in mp3_genre_mapping.items():
        # Extract just the filename from the path
        filename = os.path.basename(mp3_path)
        filename_to_genre[filename] = genre
    
    print(f"   Created filename mapping for {len(filename_to_genre)} files")
    
    for i, mp3_file in enumerate(mp3_files):
        if i >= len(fma_data['features']):
            break
            
        # Extract filename from the full path
        filename = os.path.basename(mp3_file)
        
        if filename in filename_to_genre:
            genre = filename_to_genre[filename]
            genre_index = genre_to_index[genre]
            labels.append(genre_index)
            matched_count += 1
        else:
            # Skip this sample if no genre mapping found
            skipped_count += 1
            continue
    
    print(f"   Matched: {matched_count} samples")
    print(f"   Skipped: {skipped_count} samples")
    
    # Filter features to match the labels
    if len(labels) < len(fma_data['features']):
        fma_data['features'] = fma_data['features'][:len(labels)]
    
    # Add labels to the data
    fma_data['labels'] = labels
    
    # Save the updated data
    output_file = 'mfccs/fma_13_with_labels.json'
    print(f"💾 Saving updated data to {output_file}...")
    
    with open(output_file, 'w') as f:
        json.dump(fma_data, f)
    
    # Print statistics
    print(f"\n✅ FMA data with labels created!")
    print(f"   Features: {len(fma_data['features'])}")
    print(f"   Labels: {len(fma_data['labels'])}")
    print(f"   Genres: {len(genres)}")
    
    # Show label distribution
    label_counts = Counter(labels)
    print(f"\n📊 Label distribution:")
    for genre_idx, count in sorted(label_counts.items()):
        genre_name = genres[genre_idx]
        print(f"   {genre_idx}: {genre_name} - {count} samples")
    
    return fma_data

if __name__ == "__main__":
    add_labels_to_fma_data()

