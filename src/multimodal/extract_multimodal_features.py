"""
Utility script to extract multimodal features from existing audio datasets.
Converts MFCC-only datasets to comprehensive multimodal feature datasets.
"""

import os
import sys
import json
import argparse
import logging
import time
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from multimodal.feature_extractor import MultimodalFeatureExtractor
from core.constants import GTZAN_GENRES


def extract_multimodal_features_from_audio(
    audio_dir: str,
    output_file: str,
    feature_extractor: MultimodalFeatureExtractor,
    logger: logging.Logger,
    save_interval: int = 50
) -> None:
    """
    Extract multimodal features from audio files in a directory with progressive saving.
    
    Args:
        audio_dir: Directory containing audio files
        output_file: Output JSON file path
        feature_extractor: MultimodalFeatureExtractor instance
        logger: Logger instance
        save_interval: Save progress every N files
    """
    
    # Find all audio files
    audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac'}
    audio_files = []
    
    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                audio_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(audio_files)} audio files")
    
    # Initialize data structures
    features_list = []
    labels = []
    failed_files = []
    statistics = {
        'tempo_values': [],
        'harmonic_ratios': [],
        'genre_counts': Counter(),
        'feature_shapes': {},
        'processing_times': []
    }
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize JSON file with metadata
    initial_data = {
        'metadata': {
            'total_files': len(audio_files),
            'genres': GTZAN_GENRES,
            'extraction_started': True,
            'features': [],
            'labels': [],
            'failed_files': [],
            'statistics': {}
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(initial_data, f, indent=2)
    
    # Extract features with progressive saving
    for i, audio_file in enumerate(audio_files):
        start_time = time.time()
        
        if i % 100 == 0:
            logger.info(f"Processing file {i+1}/{len(audio_files)}: {audio_file}")
        
        # Extract genre from directory structure (assuming GTZAN format)
        genre = os.path.basename(os.path.dirname(audio_file))
        if genre not in GTZAN_GENRES:
            logger.warning(f"Unknown genre: {genre} in {audio_file}")
            continue
        
        # Extract multimodal features
        multimodal_features = feature_extractor.extract_features(audio_file)
        
        if multimodal_features is not None:
            # Convert to serializable format
            features_dict = {
                # Spectral features
                'mel_spectrogram': multimodal_features.mel_spectrogram.tolist(),
                'chroma': multimodal_features.chroma.tolist(),
                'spectral_centroid': multimodal_features.spectral_centroid.tolist(),
                'spectral_rolloff': multimodal_features.spectral_rolloff.tolist(),
                'spectral_contrast': multimodal_features.spectral_contrast.tolist(),
                'zero_crossing_rate': multimodal_features.zero_crossing_rate.tolist(),
                
                # Temporal features
                'mfcc': multimodal_features.mfcc.tolist(),
                'delta_mfcc': multimodal_features.delta_mfcc.tolist(),
                'delta2_mfcc': multimodal_features.delta2_mfcc.tolist(),
                
                # Statistical features
                'tempo': float(multimodal_features.tempo.item() if hasattr(multimodal_features.tempo, 'item') else multimodal_features.tempo),
                'beat_frames': multimodal_features.beat_frames.tolist(),
                'onset_strength': multimodal_features.onset_strength.tolist(),
                'harmonic_percussive_ratio': float(multimodal_features.harmonic_percussive_ratio.item() if hasattr(multimodal_features.harmonic_percussive_ratio, 'item') else multimodal_features.harmonic_percussive_ratio),
                'spectral_bandwidth': multimodal_features.spectral_bandwidth.tolist(),
                'spectral_flatness': multimodal_features.spectral_flatness.tolist(),
            }
            
            features_list.append(features_dict)
            labels.append(genre)
            
            # Collect statistics
            statistics['tempo_values'].append(multimodal_features.tempo)
            statistics['harmonic_ratios'].append(multimodal_features.harmonic_percussive_ratio)
            statistics['genre_counts'][genre] += 1
            
            # Store feature shapes from first successful extraction
            if not statistics['feature_shapes']:
                statistics['feature_shapes'] = feature_extractor.get_feature_shapes(multimodal_features)
        else:
            failed_files.append(audio_file)
        
        # Record processing time
        processing_time = time.time() - start_time
        statistics['processing_times'].append(processing_time)
        
        # Progressive save every save_interval files
        if (i + 1) % save_interval == 0 or i == len(audio_files) - 1:
            logger.info(f"Saving progress... ({len(features_list)} successful extractions so far)")
            
            # Update the JSON file
            updated_data = {
                'metadata': {
                    'total_files': len(audio_files),
                    'processed_files': i + 1,
                    'successful_extractions': len(features_list),
                    'failed_extractions': len(failed_files),
                    'genres': GTZAN_GENRES,
                    'extraction_completed': i == len(audio_files) - 1,
                    'statistics': {
                        'genre_distribution': dict(statistics['genre_counts']),
                        'avg_processing_time': np.mean(statistics['processing_times']),
                        'tempo_stats': {
                            'mean': float(np.mean(statistics['tempo_values'])),
                            'std': float(np.std(statistics['tempo_values'])),
                            'min': float(np.min(statistics['tempo_values'])),
                            'max': float(np.max(statistics['tempo_values']))
                        },
                        'harmonic_ratio_stats': {
                            'mean': float(np.mean(statistics['harmonic_ratios'])),
                            'std': float(np.std(statistics['harmonic_ratios'])),
                            'min': float(np.min(statistics['harmonic_ratios'])),
                            'max': float(np.max(statistics['harmonic_ratios']))
                        },
                        'feature_shapes': statistics['feature_shapes']
                    }
                },
                'features': features_list,
                'labels': labels,
                'failed_files': failed_files
            }
            
            # Write to temporary file first, then rename (atomic operation)
            temp_file = output_file + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(updated_data, f, indent=2)
            os.rename(temp_file, output_file)
    
    # Generate descriptive plots and statistics
    generate_dataset_analysis(output_file, statistics, logger)
    
    logger.info(f"Multimodal features saved to {output_file}")
    logger.info(f"Successfully extracted features from {len(features_list)} files")
    logger.info(f"Failed to extract features from {len(failed_files)} files")


def generate_dataset_analysis(output_file: str, statistics: Dict, logger: logging.Logger) -> None:
    """
    Generate descriptive plots and statistics about the dataset.
    
    Args:
        output_file: Path to the output JSON file
        statistics: Dictionary containing collected statistics
        logger: Logger instance
    """
    
    # Create analysis directory
    analysis_dir = os.path.join(os.path.dirname(output_file), 'dataset_analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    logger.info("Generating dataset analysis plots...")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Genre Distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Genre count bar plot
    genre_counts = statistics['genre_counts']
    genres = list(genre_counts.keys())
    counts = list(genre_counts.values())
    
    axes[0, 0].bar(genres, counts, color=sns.color_palette("husl", len(genres)))
    axes[0, 0].set_title('Genre Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Genre')
    axes[0, 0].set_ylabel('Number of Samples')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Genre pie chart
    axes[0, 1].pie(counts, labels=genres, autopct='%1.1f%%', startangle=90)
    axes[0, 1].set_title('Genre Distribution (Percentage)', fontsize=14, fontweight='bold')
    
    # Tempo distribution
    tempo_values = np.array(statistics['tempo_values'])
    axes[1, 0].hist(tempo_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 0].set_title('Tempo Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Tempo (BPM)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].axvline(np.mean(tempo_values), color='red', linestyle='--', label=f'Mean: {np.mean(tempo_values):.1f}')
    axes[1, 0].legend()
    
    # Harmonic/Percussive ratio distribution
    harmonic_ratios = np.array(statistics['harmonic_ratios'])
    axes[1, 1].hist(harmonic_ratios, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1, 1].set_title('Harmonic/Percussive Ratio Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Harmonic/Percussive Ratio')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].axvline(np.mean(harmonic_ratios), color='red', linestyle='--', label=f'Mean: {np.mean(harmonic_ratios):.2f}')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, 'dataset_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Tempo by Genre
    plt.figure(figsize=(12, 8))
    
    # Create tempo by genre plot (we need to load the actual data for this)
    try:
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        # Extract tempo and genre data
        tempo_by_genre = {}
        for features, label in zip(data['features'], data['labels']):
            if label not in tempo_by_genre:
                tempo_by_genre[label] = []
            tempo_by_genre[label].append(features['tempo'])
        
        # Create box plot
        genre_tempo_data = []
        genre_labels = []
        for genre, tempos in tempo_by_genre.items():
            genre_tempo_data.extend(tempos)
            genre_labels.extend([genre] * len(tempos))
        
        df_tempo = pd.DataFrame({'Genre': genre_labels, 'Tempo': genre_tempo_data})
        
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df_tempo, x='Genre', y='Tempo')
        plt.title('Tempo Distribution by Genre', fontsize=16, fontweight='bold')
        plt.xlabel('Genre', fontsize=12)
        plt.ylabel('Tempo (BPM)', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'tempo_by_genre.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logger.warning(f"Could not create tempo by genre plot: {e}")
    
    # 3. Processing Time Analysis
    processing_times = np.array(statistics['processing_times'])
    
    plt.figure(figsize=(10, 6))
    plt.plot(processing_times, alpha=0.7, color='purple')
    plt.title('Processing Time per File', fontsize=14, fontweight='bold')
    plt.xlabel('File Index')
    plt.ylabel('Processing Time (seconds)')
    plt.axhline(np.mean(processing_times), color='red', linestyle='--', 
                label=f'Average: {np.mean(processing_times):.3f}s')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, 'processing_times.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Feature Shapes Summary
    feature_shapes = statistics['feature_shapes']
    
    plt.figure(figsize=(12, 8))
    feature_names = list(feature_shapes.keys())
    feature_dims = [np.prod(shape) for shape in feature_shapes.values()]
    
    bars = plt.bar(range(len(feature_names)), feature_dims, color=sns.color_palette("Set3", len(feature_names)))
    plt.title('Feature Dimensions', fontsize=14, fontweight='bold')
    plt.xlabel('Feature Type')
    plt.ylabel('Total Dimensions')
    plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, dim in zip(bars, feature_dims):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(feature_dims)*0.01,
                f'{dim}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, 'feature_dimensions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Generate summary statistics text file
    summary_file = os.path.join(analysis_dir, 'dataset_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("MULTIMODAL DATASET ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total Files Processed: {sum(statistics['genre_counts'].values())}\n")
        f.write(f"Successful Extractions: {len(statistics['tempo_values'])}\n")
        f.write(f"Failed Extractions: {len(statistics['processing_times']) - len(statistics['tempo_values'])}\n")
        f.write(f"Success Rate: {len(statistics['tempo_values'])/len(statistics['processing_times'])*100:.1f}%\n\n")
        
        f.write("GENRE DISTRIBUTION:\n")
        f.write("-" * 20 + "\n")
        for genre, count in sorted(statistics['genre_counts'].items()):
            f.write(f"{genre:12}: {count:3d} samples ({count/len(statistics['tempo_values'])*100:.1f}%)\n")
        
        f.write(f"\nTEMPO STATISTICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Mean: {np.mean(tempo_values):.1f} BPM\n")
        f.write(f"Std:  {np.std(tempo_values):.1f} BPM\n")
        f.write(f"Min:  {np.min(tempo_values):.1f} BPM\n")
        f.write(f"Max:  {np.max(tempo_values):.1f} BPM\n")
        
        f.write(f"\nHARMONIC/PERCUSSIVE RATIO STATISTICS:\n")
        f.write("-" * 35 + "\n")
        f.write(f"Mean: {np.mean(harmonic_ratios):.3f}\n")
        f.write(f"Std:  {np.std(harmonic_ratios):.3f}\n")
        f.write(f"Min:  {np.min(harmonic_ratios):.3f}\n")
        f.write(f"Max:  {np.max(harmonic_ratios):.3f}\n")
        
        f.write(f"\nPROCESSING STATISTICS:\n")
        f.write("-" * 25 + "\n")
        f.write(f"Average Processing Time: {np.mean(processing_times):.3f} seconds\n")
        f.write(f"Total Processing Time: {np.sum(processing_times):.1f} seconds\n")
        f.write(f"Files per Minute: {len(processing_times)/(np.sum(processing_times)/60):.1f}\n")
        
        f.write(f"\nFEATURE SHAPES:\n")
        f.write("-" * 15 + "\n")
        for feature_name, shape in feature_shapes.items():
            f.write(f"{feature_name:25}: {shape}\n")
    
    logger.info(f"Dataset analysis saved to {analysis_dir}")
    logger.info(f"Generated plots: dataset_overview.png, tempo_by_genre.png, processing_times.png, feature_dimensions.png")
    logger.info(f"Generated summary: dataset_summary.txt")


def convert_mfcc_to_multimodal(
    mfcc_data_file: str,
    output_file: str,
    feature_extractor: MultimodalFeatureExtractor,
    logger: logging.Logger
) -> None:
    """
    Convert existing MFCC dataset to multimodal features.
    This is a placeholder - in practice, you'd need the original audio files.
    
    Args:
        mfcc_data_file: Path to existing MFCC JSON file
        output_file: Output multimodal JSON file path
        feature_extractor: MultimodalFeatureExtractor instance
        logger: Logger instance
    """
    
    logger.warning("Converting MFCC to multimodal features requires original audio files.")
    logger.warning("This function is a placeholder - you need to extract from audio directly.")
    
    # Load existing MFCC data
    with open(mfcc_data_file, 'r') as f:
        mfcc_data = json.load(f)
    
    logger.info(f"Loaded MFCC data with {len(mfcc_data['features'])} samples")
    
    # For demonstration, create dummy multimodal features
    # In practice, you'd need to re-extract from original audio files
    multimodal_features_list = []
    
    for i, (mfcc_features, label) in enumerate(zip(mfcc_data['features'], mfcc_data['labels'])):
        if i % 100 == 0:
            logger.info(f"Converting sample {i+1}/{len(mfcc_data['features'])}")
        
        # Create dummy multimodal features based on MFCC
        # This is just for demonstration - real implementation needs audio files
        mfcc_array = np.array(mfcc_features)
        
        # Create dummy multimodal features
        multimodal_features_dict = {
            # Spectral features (dummy - would need audio)
            'mel_spectrogram': np.random.randn(128, mfcc_array.shape[1]).tolist(),
            'chroma': np.random.randn(12, mfcc_array.shape[1]).tolist(),
            'spectral_centroid': np.random.randn(1, mfcc_array.shape[1]).tolist(),
            'spectral_rolloff': np.random.randn(1, mfcc_array.shape[1]).tolist(),
            'spectral_contrast': np.random.randn(7, mfcc_array.shape[1]).tolist(),
            'zero_crossing_rate': np.random.randn(1, mfcc_array.shape[1]).tolist(),
            
            # Temporal features (use actual MFCC)
            'mfcc': mfcc_array[:13].tolist(),
            'delta_mfcc': np.random.randn(13, mfcc_array.shape[1]).tolist(),
            'delta2_mfcc': np.random.randn(13, mfcc_array.shape[1]).tolist(),
            
            # Statistical features (dummy - would need audio)
            'tempo': float(np.random.uniform(60, 180)),
            'beat_frames': np.random.randint(0, mfcc_array.shape[1], 10).tolist(),
            'onset_strength': np.random.randn(100).tolist(),
            'harmonic_percussive_ratio': float(np.random.uniform(0.1, 2.0)),
            'spectral_bandwidth': np.random.randn(100).tolist(),
            'spectral_flatness': np.random.randn(100).tolist(),
        }
        
        multimodal_features_list.append(multimodal_features_dict)
    
    # Create output data
    output_data = {
        'features': multimodal_features_list,
        'labels': mfcc_data['labels'],
        'genres': mfcc_data.get('genres', GTZAN_GENRES),
        'note': 'This is dummy multimodal data created from MFCC. For real multimodal features, extract from original audio files.',
        'original_mfcc_file': mfcc_data_file
    }
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Dummy multimodal features saved to {output_file}")
    logger.warning("This contains dummy spectral and statistical features. Use original audio files for real multimodal extraction.")


def main():
    """Main function for multimodal feature extraction."""
    parser = argparse.ArgumentParser(description='Extract multimodal features from audio data')
    parser.add_argument('--input', required=True, help='Input audio directory or MFCC JSON file')
    parser.add_argument('--output', required=True, help='Output multimodal JSON file')
    parser.add_argument('--mode', choices=['audio', 'mfcc'], default='audio', 
                       help='Input mode: audio directory or MFCC JSON file')
    parser.add_argument('--n-mfcc', type=int, default=13, help='Number of MFCC coefficients')
    parser.add_argument('--n-mels', type=int, default=128, help='Number of mel bands')
    parser.add_argument('--save-interval', type=int, default=50, help='Save progress every N files')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize feature extractor
    feature_extractor = MultimodalFeatureExtractor(n_mfcc=args.n_mfcc, n_mels=args.n_mels)
    
    if args.mode == 'audio':
        # Extract from audio files
        if not os.path.isdir(args.input):
            logger.error(f"Input directory does not exist: {args.input}")
            return
        
        extract_multimodal_features_from_audio(
            args.input, args.output, feature_extractor, logger, args.save_interval
        )
    
    elif args.mode == 'mfcc':
        # Convert from MFCC data
        if not os.path.isfile(args.input):
            logger.error(f"Input MFCC file does not exist: {args.input}")
            return
        
        convert_mfcc_to_multimodal(
            args.input, args.output, feature_extractor, logger
        )
    
    logger.info("Multimodal feature extraction completed!")


if __name__ == "__main__":
    main()
