#!/usr/bin/env python3
"""
Extract autoencoded features and save in the same format as MFCC extractions.

This creates JSON files with autoencoded latent features that can be plugged
directly into the existing training framework, replacing MFCC features.
"""

import os
import sys
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
from pathlib import Path
from typing import List, Tuple, Dict, Any
import argparse

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from autoencoder import SongLevelAutoencoderExtractor, SongLevelAutoencoder
from core.config import AudioConfig
from core.constants import SAMPLE_RATE


def extract_song_level_features_json(gtzan_path: str, output_file: str, 
                                   latent_dim: int = 256, epochs: int = 20,
                                   batch_size: int = 4, song_length: float = 30.0,
                                   checkpoint_file: str = None, resume: bool = False) -> str:
    """
    Extract song-level autoencoded features and save in MFCC-compatible JSON format.
    Supports incremental saving and checkpoint resuming.
    
    Args:
        gtzan_path: Path to GTZAN dataset
        output_file: Output JSON file path
        latent_dim: Latent dimension for autoencoder
        epochs: Training epochs for autoencoder
        batch_size: Batch size for training
        song_length: Length of songs in seconds
        checkpoint_file: Path to checkpoint file for resuming
        resume: Whether to resume from checkpoint
        
    Returns:
        Path to the created JSON file
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🎵 Extracting Song-Level Autoencoded Features (Incremental)")
    logger.info("=" * 50)
    
    # Create audio config
    config = AudioConfig()
    config.sample_rate = SAMPLE_RATE
    config.max_duration = song_length
    
    # Get GTZAN audio files
    audio_files = []
    for genre in sorted(os.listdir(gtzan_path)):
        genre_path = os.path.join(gtzan_path, genre)
        if os.path.isdir(genre_path):
            for file in sorted(os.listdir(genre_path)):
                if file.endswith('.wav'):
                    audio_files.append((os.path.join(genre_path, file), genre))
    
    logger.info(f"Found {len(audio_files)} songs")
    logger.info(f"Genres: {sorted(set(genre for _, genre in audio_files))}")
    
    # Setup checkpoint file
    if checkpoint_file is None:
        checkpoint_file = output_file.replace('.json', '_checkpoint.json')
    
    # Initialize data structures
    all_encodings = []
    all_labels = []
    all_genres = []
    start_idx = 0
    
    # Resume from checkpoint if requested
    if resume and os.path.exists(checkpoint_file):
        logger.info(f"📂 Resuming from checkpoint: {checkpoint_file}")
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            all_encodings = checkpoint_data.get('encodings', [])
            all_labels = checkpoint_data.get('labels', [])
            all_genres = checkpoint_data.get('genres', [])
            start_idx = len(all_encodings)
            logger.info(f"✅ Resumed: {start_idx} songs already processed")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting from beginning.")
            start_idx = 0
    else:
        logger.info("🚀 Starting fresh extraction")
    
    # Initialize song-level extractor
    extractor = SongLevelAutoencoderExtractor(
        config=config,
        latent_dim=latent_dim,
        song_length=song_length,
        logger=logger
    )
    
    # Force CPU due to GPU memory constraints
    logger.warning("Using CPU due to GPU memory constraints")
    extractor.device = torch.device('cpu')
    
    # Process songs incrementally
    logger.info(f"Processing songs {start_idx + 1} to {len(audio_files)}...")
    
    for song_idx in range(start_idx, len(audio_files)):
        file_path, genre = audio_files[song_idx]
        logger.info(f"Training FRESH autoencoder on song {song_idx + 1}/{len(audio_files)}: {os.path.basename(file_path)} ({genre})")
        
        try:
            # Create fresh autoencoder for this song
            fresh_autoencoder = SongLevelAutoencoder(
                song_length=int(config.sample_rate * song_length),
                latent_dim=latent_dim
            )
            fresh_autoencoder = fresh_autoencoder.to(extractor.device)
            
            # Load and preprocess single song
            audio, sr = librosa.load(file_path, sr=config.sample_rate, duration=song_length)
            if len(audio) < config.sample_rate * song_length * 0.5:
                logger.warning(f"Skipping {file_path}: too short")
                continue
                
            # Normalize
            audio = (audio - audio.mean()) / (audio.std() + 1e-8)
            
            # Convert to tensor
            audio_tensor = torch.FloatTensor(audio).unsqueeze(0).unsqueeze(0)
            audio_tensor = audio_tensor.to(extractor.device)
            
            # Train this fresh autoencoder
            criterion = nn.MSELoss()
            optimizer = optim.Adam(fresh_autoencoder.parameters(), lr=1e-3)
            
            song_losses = []
            for epoch in range(epochs):
                fresh_autoencoder.train()
                optimizer.zero_grad()
                
                reconstructed, latent = fresh_autoencoder(audio_tensor)
                loss = criterion(reconstructed, audio_tensor)
                loss.backward()
                optimizer.step()
                
                song_losses.append(loss.item())
                
                if epoch % 2 == 0:
                    logger.info(f"Song {song_idx + 1}, Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6f}")
            
            # Extract final encoding
            with torch.no_grad():
                fresh_autoencoder.eval()
                _, final_encoding = fresh_autoencoder(audio_tensor)
                final_encoding = final_encoding.cpu().numpy().flatten()
            
            # Store results
            all_encodings.append(final_encoding.tolist())
            all_labels.append(genre)
            all_genres.append(genre)
            
            avg_loss = np.mean(song_losses)
            logger.info(f"Song {song_idx + 1} completed. Avg Loss: {avg_loss:.6f}, Encoding shape: {final_encoding.shape}")
            
            # Save incremental checkpoint every 10 songs
            if (song_idx + 1) % 10 == 0:
                checkpoint_data = {
                    'encodings': all_encodings,
                    'labels': all_labels,
                    'genres': all_genres,
                    'processed_songs': song_idx + 1,
                    'total_songs': len(audio_files),
                    'config': {
                        'latent_dim': latent_dim,
                        'sample_rate': config.sample_rate,
                        'song_length': song_length
                    }
                }
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
                logger.info(f"💾 Checkpoint saved: {song_idx + 1}/{len(audio_files)} songs processed")
            
            # Clean up
            del fresh_autoencoder
            torch.cuda.empty_cache() if extractor.device.type == 'cuda' else None
            
        except Exception as e:
            logger.warning(f"Failed to process {file_path}: {e}")
            continue
    
    # Create final output
    logger.info("Creating final MFCC-compatible JSON format...")
    
    # Encode genre labels to integers
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(all_labels)
    unique_genres = label_encoder.classes_.tolist()
    
    # Reshape encodings to match MFCC format
    encodings_array = np.array(all_encodings)
    features_reshaped = encodings_array.reshape(encodings_array.shape[0], 1, encodings_array.shape[1])
    
    # Create final output data
    output_data = {
        'features': features_reshaped.tolist(),
        'labels': labels_encoded.tolist(),
        'mapping': unique_genres,
        'config': {
            'latent_dim': latent_dim,
            'sample_rate': config.sample_rate,
            'song_length': song_length,
            'num_songs': len(all_encodings),
            'feature_type': 'fresh_song_level_autoencoded',
            'description': f'Fresh song-level autoencoded features with {latent_dim}D latent space (one autoencoder per song)'
        }
    }
    
    # Save final JSON file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Clean up checkpoint file
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        logger.info(f"🗑️ Checkpoint file cleaned up: {checkpoint_file}")
    
    logger.info(f"✅ Song-level features saved to {output_file}")
    logger.info(f"Features shape: {features_reshaped.shape}")
    logger.info(f"Format: (samples, frames, features) = ({features_reshaped.shape[0]}, {features_reshaped.shape[1]}, {features_reshaped.shape[2]})")
    
    return output_file


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Extract fresh autoencoded features in MFCC-compatible format")
    parser.add_argument("--gtzan-path", required=True, help="Path to GTZAN dataset")
    parser.add_argument("--output-file", required=True, help="Output JSON file path")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent dimension")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs per song")
    parser.add_argument("--song-length", type=float, default=30.0, help="Song length in seconds")
    parser.add_argument("--checkpoint-file", help="Path to checkpoint file for resuming")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    extract_song_level_features_json(
        gtzan_path=args.gtzan_path,
        output_file=args.output_file,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        song_length=args.song_length,
        checkpoint_file=args.checkpoint_file,
        resume=args.resume
    )


if __name__ == "__main__":
    main()
