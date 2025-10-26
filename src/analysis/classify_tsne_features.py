#!/usr/bin/env python3
"""
Classification Test for t-SNE Compressed Features

This script tests how well the t-SNE compressed features can be classified
by training various classifiers on the 2D coordinates.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import argparse
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def load_mfcc_data(json_file):
    """Load MFCC features from JSON file."""
    print(f"Loading MFCC data from: {json_file}")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    features = data['features']
    labels = data['labels']
    
    # Flatten each song to a single vector by taking mean across frames
    features_array = np.array([np.mean(song, axis=0) for song in features])
    labels_array = np.array(labels)
    
    print(f"Loaded {len(features_array)} songs")
    print(f"Feature shape: {features_array.shape}")
    print(f"Unique genres: {sorted(set(labels_array))}")
    
    return features_array, labels_array, data.get('mapping', sorted(set(labels_array)))

def load_tsne_coordinates(tsne_file):
    """Load t-SNE coordinates from a saved file."""
    if os.path.exists(tsne_file):
        print(f"Loading t-SNE coordinates from: {tsne_file}")
        data = np.load(tsne_file)
        return data
    else:
        print(f"t-SNE file not found: {tsne_file}")
        return None

def run_classification_tests(X_original, X_tsne, y, mapping, output_dir):
    """Run comprehensive classification tests."""
    print("\n=== Classification Tests ===")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', random_state=42),
        'SVM (Linear)': SVC(kernel='linear', random_state=42),
        'k-NN (k=3)': KNeighborsClassifier(n_neighbors=3),
        'k-NN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'k-NN (k=7)': KNeighborsClassifier(n_neighbors=7),
    }
    
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Results storage
    results = {
        'original': {},
        'tsne': {}
    }
    
    # Test on original features
    print("\n--- Testing Original 13D MFCC Features ---")
    scaler_orig = StandardScaler()
    X_orig_scaled = scaler_orig.fit_transform(X_original)
    
    for name, clf in classifiers.items():
        scores = cross_val_score(clf, X_orig_scaled, y, cv=cv, scoring='accuracy')
        mean_score = scores.mean()
        std_score = scores.std()
        results['original'][name] = {'mean': mean_score, 'std': std_score}
        print(f"{name:20}: {mean_score:.4f} ± {std_score:.4f}")
    
    # Test on t-SNE features
    print("\n--- Testing 2D t-SNE Features ---")
    scaler_tsne = StandardScaler()
    X_tsne_scaled = scaler_tsne.fit_transform(X_tsne)
    
    for name, clf in classifiers.items():
        scores = cross_val_score(clf, X_tsne_scaled, y, cv=cv, scoring='accuracy')
        mean_score = scores.mean()
        std_score = scores.std()
        results['tsne'][name] = {'mean': mean_score, 'std': std_score}
        print(f"{name:20}: {mean_score:.4f} ± {std_score:.4f}")
    
    # Create comparison plots
    create_comparison_plots(results, mapping, output_dir)
    
    # Detailed analysis with best classifier
    best_orig = max(results['original'].items(), key=lambda x: x[1]['mean'])
    best_tsne = max(results['tsne'].items(), key=lambda x: x[1]['mean'])
    
    print(f"\n--- Best Classifiers ---")
    print(f"Original features: {best_orig[0]} ({best_orig[1]['mean']:.4f})")
    print(f"t-SNE features: {best_tsne[0]} ({best_tsne[1]['mean']:.4f})")
    
    # Information loss analysis
    info_loss = best_orig[1]['mean'] - best_tsne[1]['mean']
    print(f"Information loss: {info_loss:.4f} ({info_loss/best_orig[1]['mean']*100:.1f}%)")
    
    return results

def create_comparison_plots(results, mapping, output_dir):
    """Create comparison plots for classification results."""
    print("Creating comparison plots...")
    
    # Extract data for plotting
    classifiers = list(results['original'].keys())
    orig_means = [results['original'][clf]['mean'] for clf in classifiers]
    orig_stds = [results['original'][clf]['std'] for clf in classifiers]
    tsne_means = [results['tsne'][clf]['mean'] for clf in classifiers]
    tsne_stds = [results['tsne'][clf]['std'] for clf in classifiers]
    
    # 1. Bar plot comparison
    plt.figure(figsize=(15, 8))
    x = np.arange(len(classifiers))
    width = 0.35
    
    plt.bar(x - width/2, orig_means, width, yerr=orig_stds, 
            label='Original 13D MFCC', alpha=0.8, capsize=5)
    plt.bar(x + width/2, tsne_means, width, yerr=tsne_stds, 
            label='2D t-SNE', alpha=0.8, capsize=5)
    
    plt.xlabel('Classifier', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Classification Performance: Original vs t-SNE Features', fontsize=14)
    plt.xticks(x, classifiers, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, 'classification_comparison.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_file}")
    plt.close()
    
    # 2. Scatter plot: Original vs t-SNE accuracy
    plt.figure(figsize=(10, 8))
    plt.scatter(orig_means, tsne_means, s=100, alpha=0.7)
    
    # Add diagonal line (no information loss)
    min_acc = min(min(orig_means), min(tsne_means))
    max_acc = max(max(orig_means), max(tsne_means))
    plt.plot([min_acc, max_acc], [min_acc, max_acc], 'r--', alpha=0.5, label='No information loss')
    
    # Add classifier labels
    for i, clf in enumerate(classifiers):
        plt.annotate(clf, (orig_means[i], tsne_means[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('Original 13D MFCC Accuracy', fontsize=12)
    plt.ylabel('2D t-SNE Accuracy', fontsize=12)
    plt.title('Information Loss in t-SNE Compression', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    scatter_file = os.path.join(output_dir, 'information_loss_scatter.png')
    plt.savefig(scatter_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {scatter_file}")
    plt.close()

def analyze_neighborhood_structure(X_tsne, y, mapping, output_dir):
    """Analyze the local neighborhood structure in t-SNE space."""
    print("\n=== Neighborhood Structure Analysis ===")
    
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import adjusted_rand_score
    
    # Find k-nearest neighbors for each point
    k_values = [3, 5, 7, 10, 15, 20]
    neighborhood_scores = []
    
    for k in k_values:
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(X_tsne)  # +1 to exclude self
        distances, indices = nbrs.kneighbors(X_tsne)
        
        # Calculate neighborhood purity (fraction of neighbors with same label)
        purities = []
        for i in range(len(X_tsne)):
            neighbor_labels = y[indices[i, 1:]]  # Exclude self
            same_label_count = np.sum(neighbor_labels == y[i])
            purity = same_label_count / k
            purities.append(purity)
        
        avg_purity = np.mean(purities)
        neighborhood_scores.append(avg_purity)
        print(f"k={k:2d}: Average neighborhood purity = {avg_purity:.4f}")
    
    # Plot neighborhood purity
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, neighborhood_scores, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Neighbors (k)', fontsize=12)
    plt.ylabel('Average Neighborhood Purity', fontsize=12)
    plt.title('Local Neighborhood Structure in t-SNE Space', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    purity_file = os.path.join(output_dir, 'neighborhood_purity.png')
    plt.savefig(purity_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {purity_file}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Classification Test for t-SNE Features')
    parser.add_argument('--input', '-i', required=True, 
                       help='Path to MFCC JSON file')
    parser.add_argument('--tsne-coords', '-t', 
                       help='Path to saved t-SNE coordinates (optional)')
    parser.add_argument('--output-dir', '-o', default='outputs/classification_test',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    try:
        # Load original data
        X_original, y, mapping = load_mfcc_data(args.input)
        
        # Load or compute t-SNE coordinates
        if args.tsne_coords and os.path.exists(args.tsne_coords):
            X_tsne = load_tsne_coordinates(args.tsne_coords)
        else:
            print("Computing t-SNE coordinates...")
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, random_state=42, verbose=1)
            X_tsne = tsne.fit_transform(X_original)
        
        if X_tsne is None:
            print("Error: Could not load or compute t-SNE coordinates")
            return 1
        
        print(f"t-SNE coordinates shape: {X_tsne.shape}")
        
        # Run classification tests
        results = run_classification_tests(X_original, X_tsne, y, mapping, args.output_dir)
        
        # Analyze neighborhood structure
        analyze_neighborhood_structure(X_tsne, y, mapping, args.output_dir)
        
        print(f"\n✅ Classification analysis complete! Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
