#!/usr/bin/env python3
"""
t-SNE Analysis for Autoencoded Features

This script performs t-SNE dimensionality reduction on autoencoded features
to visualize how well the features cluster by genre.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import argparse
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def load_autoencoded_data(json_file):
    """Load autoencoded features from JSON file."""
    print(f"Loading data from: {json_file}")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract features and labels
    features = []
    labels = []
    
    for i, song_features in enumerate(data['features']):
        # Each song has shape: [[feat1, feat2, ..., feat16]]
        # We want just the 16D vector
        if len(song_features) == 1 and len(song_features[0]) > 0:
            features.append(song_features[0])  # Extract the 16D vector
            labels.append(data['labels'][i])
        else:
            print(f"Warning: Skipping song {i} with unexpected feature shape: {song_features}")
    
    features = np.array(features)
    labels = np.array(labels)
    
    print(f"Loaded {len(features)} songs")
    print(f"Feature shape: {features.shape}")
    print(f"Unique genres: {sorted(set(labels))}")
    
    return features, labels, data.get('mapping', sorted(set(labels)))

def load_mfcc_data(json_file):
    """Load MFCC features from JSON file."""
    print(f"Loading MFCC data from: {json_file}")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    features = data['features']
    labels = data['labels']
    
    print(f"Loaded {len(features)} songs")
    print(f"First feature shape: {len(features[0]) if features else 'Empty'}")
    print(f"First feature[0] shape: {len(features[0][0]) if features and features[0] else 'Empty'}")
    print(f"Unique genres: {sorted(set(labels))}")
    
    # MFCC format: List of songs, each song is list of frames, each frame is list of coefficients
    # Flatten each song to a single vector by taking mean across frames
    features_array = np.array([np.mean(song, axis=0) for song in features])
    print(f"Flattened feature shape: {features_array.shape}")
    
    labels_array = np.array(labels)
    
    return features_array, labels_array, data.get('mapping', sorted(set(labels)))

def analyze_feature_statistics(features, labels):
    """Analyze basic statistics of the features."""
    print("\n=== Feature Statistics ===")
    print(f"Feature shape: {features.shape}")
    print(f"Min value: {features.min():.6f}")
    print(f"Max value: {features.max():.6f}")
    print(f"Mean value: {features.mean():.6f}")
    print(f"Std value: {features.std():.6f}")
    
    # Sparsity analysis
    zero_count = np.sum(features == 0)
    total_elements = features.size
    sparsity = zero_count / total_elements * 100
    print(f"Sparsity: {sparsity:.2f}% ({zero_count}/{total_elements} zeros)")
    
    # Per-genre statistics
    print("\n=== Per-Genre Statistics ===")
    unique_genres = sorted(set(labels))
    for genre in unique_genres:
        genre_mask = labels == genre
        genre_features = features[genre_mask]
        print(f"{genre:12}: {np.sum(genre_mask):3d} songs, "
              f"mean={genre_features.mean():.4f}, "
              f"std={genre_features.std():.4f}, "
              f"sparsity={np.sum(genre_features == 0) / genre_features.size * 100:.1f}%")

def run_tsne_analysis(features, labels, mapping, output_dir, perplexity=30, n_iter=1000):
    """Run t-SNE analysis and create visualizations."""
    print(f"\n=== Running t-SNE Analysis ===")
    print(f"Perplexity: {perplexity}")
    print(f"Iterations: {n_iter}")
    
    # Encode labels as integers
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    # Run t-SNE
    print("Computing t-SNE embedding...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,
        random_state=42,
        verbose=1
    )
    
    features_2d = tsne.fit_transform(features)
    
    # Create visualizations
    create_tsne_plots(features_2d, labels, mapping, output_dir)
    
    return features_2d, encoded_labels

def create_tsne_plots(features_2d, labels, mapping, output_dir):
    """Create various t-SNE visualization plots."""
    print("Creating t-SNE visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Basic t-SNE plot with genre colors
    plt.figure(figsize=(12, 8))
    unique_genres = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_genres)))
    
    for i, genre in enumerate(unique_genres):
        mask = labels == genre
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=[colors[i]], label=genre, alpha=0.7, s=50)
    
    plt.title('t-SNE Visualization of Music Features', fontsize=16)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, 'tsne_genre_clusters.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_file}")
    plt.close()
    
    # 2. Density plot
    plt.figure(figsize=(10, 8))
    plt.hist2d(features_2d[:, 0], features_2d[:, 1], bins=50, cmap='Blues')
    plt.colorbar(label='Density')
    plt.title('t-SNE Density Plot', fontsize=16)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.tight_layout()
    
    density_file = os.path.join(output_dir, 'tsne_density.png')
    plt.savefig(density_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {density_file}")
    plt.close()
    
    # 3. Per-genre subplots
    n_genres = len(unique_genres)
    n_cols = 3
    n_rows = (n_genres + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, genre in enumerate(unique_genres):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        mask = labels == genre
        ax.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                  c=colors[i], alpha=0.7, s=50)
        ax.set_title(f'{genre} ({np.sum(mask)} songs)', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_genres, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.suptitle('t-SNE Visualization by Genre', fontsize=16)
    plt.tight_layout()
    
    subplot_file = os.path.join(output_dir, 'tsne_by_genre.png')
    plt.savefig(subplot_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {subplot_file}")
    plt.close()

def analyze_clustering_quality(features_2d, labels):
    """Analyze the quality of clustering in t-SNE space."""
    print("\n=== Clustering Quality Analysis ===")
    
    # Calculate within-genre and between-genre distances
    unique_genres = sorted(set(labels))
    
    within_genre_distances = []
    between_genre_distances = []
    
    for genre in unique_genres:
        genre_mask = labels == genre
        genre_points = features_2d[genre_mask]
        
        if len(genre_points) > 1:
            # Within-genre distances
            for i in range(len(genre_points)):
                for j in range(i+1, len(genre_points)):
                    dist = np.linalg.norm(genre_points[i] - genre_points[j])
                    within_genre_distances.append(dist)
        
        # Between-genre distances
        other_genres = [g for g in unique_genres if g != genre]
        for other_genre in other_genres:
            other_mask = labels == other_genre
            other_points = features_2d[other_mask]
            
            for point1 in genre_points:
                for point2 in other_points:
                    dist = np.linalg.norm(point1 - point2)
                    between_genre_distances.append(dist)
    
    if within_genre_distances and between_genre_distances:
        avg_within = np.mean(within_genre_distances)
        avg_between = np.mean(between_genre_distances)
        separation_ratio = avg_between / avg_within
        
        print(f"Average within-genre distance: {avg_within:.4f}")
        print(f"Average between-genre distance: {avg_between:.4f}")
        print(f"Separation ratio: {separation_ratio:.4f}")
        
        if separation_ratio > 2.0:
            print("✅ Good clustering: Genres are well separated")
        elif separation_ratio > 1.5:
            print("⚠️  Moderate clustering: Some genre separation")
        else:
            print("❌ Poor clustering: Genres are mixed together")

def main():
    parser = argparse.ArgumentParser(description='t-SNE Analysis for Autoencoded Features')
    parser.add_argument('--input', '-i', required=True, 
                       help='Path to autoencoded JSON file')
    parser.add_argument('--output-dir', '-o', default='outputs/tsne_analysis',
                       help='Output directory for plots')
    parser.add_argument('--perplexity', '-p', type=int, default=30,
                       help='t-SNE perplexity parameter')
    parser.add_argument('--iterations', '-n', type=int, default=1000,
                       help='Number of t-SNE iterations')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    try:
        # Detect data type and load accordingly
        with open(args.input, 'r') as f:
            data = json.load(f)
        
        # Check if it's MFCC data (nested structure with frames)
        if data['features'] and isinstance(data['features'][0][0][0], (int, float)):
            print("Detected MFCC data format")
            features, labels, mapping = load_mfcc_data(args.input)
        else:
            print("Detected autoencoded data format")
            features, labels, mapping = load_autoencoded_data(args.input)
        
        if len(features) == 0:
            print("Error: No valid features found in the input file")
            return 1
        
        # Analyze feature statistics
        analyze_feature_statistics(features, labels)
        
        # Run t-SNE analysis
        features_2d, encoded_labels = run_tsne_analysis(
            features, labels, mapping, args.output_dir, 
            args.perplexity, args.iterations
        )
        
        # Analyze clustering quality
        analyze_clustering_quality(features_2d, labels)
        
        print(f"\n✅ t-SNE analysis complete! Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
