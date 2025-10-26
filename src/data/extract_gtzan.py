#!/usr/bin/env python3
"""
Unified GTZAN Feature Extraction

This script extracts features from the GTZAN dataset using various approaches:
- MFCC features (traditional)
- Autoencoded features (CNN-based)
- Recurrent features (RNN-based)

All approaches output features in the same format for compatibility with existing training.
"""

import os
import sys
import argparse
import logging
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import librosa
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from autoencoder import SongLevelAutoencoderExtractor, RecurrentAutoencoderExtractor, SongLevelAutoencoder, RecurrentAutoencoder
from core.config import AudioConfig
from core.constants import SAMPLE_RATE


def save_training_plots(song_losses: list, output_dir: str, song_name: str, approach: str):
    """Save training loss plots for the first song to monitor training health."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(song_losses, 'b-', linewidth=2)
    plt.title(f'{approach.upper()} Autoencoder Training - {song_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Reconstruction Loss (MSE)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, f'{approach}_autoencoder_training_plot.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create detailed plot with early stopping info
    plt.figure(figsize=(12, 8))
    
    # Main loss plot
    plt.subplot(2, 1, 1)
    plt.plot(song_losses, 'b-', linewidth=2, label='Training Loss')
    plt.title(f'{approach.upper()} Autoencoder Training - {song_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Reconstruction Loss (MSE)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Loss improvement plot
    plt.subplot(2, 1, 2)
    if len(song_losses) > 1:
        improvements = [song_losses[i] - song_losses[i-1] for i in range(1, len(song_losses))]
        plt.plot(range(1, len(song_losses)), improvements, 'r-', linewidth=2, label='Loss Improvement')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.title('Loss Improvement per Epoch', fontsize=12, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss Improvement', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    
    # Save detailed plot
    detailed_plot_file = os.path.join(output_dir, f'{approach}_autoencoder_detailed_plot.png')
    plt.savefig(detailed_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_file, detailed_plot_file


def extract_mfcc_features(gtzan_path: str, output_file: str, 
                         n_mfcc: int = 13, hop_length: int = 512,
                         song_length: float = 30.0) -> str:
    """Extract traditional MFCC features from GTZAN dataset."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Extracting MFCC features from {gtzan_path}")
    logger.info(f"MFCC parameters: n_mfcc={n_mfcc}, hop_length={hop_length}")
    
    # Get GTZAN audio files
    audio_files = []
    for genre in os.listdir(gtzan_path):
        genre_path = os.path.join(gtzan_path, genre)
        if os.path.isdir(genre_path):
            for file in os.listdir(genre_path):
                if file.endswith('.wav'):
                    audio_files.append((os.path.join(genre_path, file), genre))
    
    logger.info(f"Found {len(audio_files)} songs")
    
    # Extract MFCC features
    all_features = []
    all_labels = []
    all_genres = []
    
    for file_path, genre in audio_files:
        try:
            # Load audio
            audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=song_length)
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(
                y=audio, 
                sr=sr, 
                n_mfcc=n_mfcc, 
                hop_length=hop_length
            )
            
            # Transpose to (time, features) format
            mfccs = mfccs.T
            
            all_features.append(mfccs.tolist())
            all_labels.append(genre)
            all_genres.append(genre)
            
        except Exception as e:
            logger.warning(f"Failed to process {file_path}: {e}")
            continue
    
    # Create label mapping
    unique_genres = sorted(list(set(all_genres)))
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(all_labels)
    
    # Create output data
    output_data = {
        'features': all_features,
        'labels': encoded_labels.tolist(),
        'mapping': unique_genres,
        'config': {
            'n_mfcc': n_mfcc,
            'hop_length': hop_length,
            'sample_rate': SAMPLE_RATE,
            'song_length': song_length,
            'num_songs': len(all_features),
            'extraction_type': 'mfcc'
        }
    }
    
    # Save features
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"MFCC features saved to {output_file}")
    logger.info(f"Features shape: {len(all_features)} songs")
    logger.info(f"Genres: {unique_genres}")
    
    return output_file


def extract_autoencoded_features(gtzan_path: str, output_file: str, 
                                latent_dim: int = 64, epochs: int = 1000,
                                song_length: float = 30.0, checkpoint_file: str = None,
                                resume: bool = False) -> str:
    """Extract CNN-based autoencoded features from GTZAN dataset."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Extracting CNN-based autoencoded features from {gtzan_path}")
    logger.info(f"Latent dimension: {latent_dim}, Max epochs: {epochs}")
    
    # Create audio config
    config = AudioConfig()
    config.sample_rate = SAMPLE_RATE
    config.max_duration = song_length
    
    # Get GTZAN audio files
    audio_files = []
    for genre in os.listdir(gtzan_path):
        genre_path = os.path.join(gtzan_path, genre)
        if os.path.isdir(genre_path):
            for file in os.listdir(genre_path):
                if file.endswith('.wav'):
                    audio_files.append((os.path.join(genre_path, file), genre))
    
    logger.info(f"Found {len(audio_files)} songs")
    
    # Initialize extractor
    extractor = SongLevelAutoencoderExtractor(
        config=config,
        latent_dim=latent_dim,
        song_length=song_length,
        logger=logger
    )
    
    # Initialize output file with empty structure
    output_data = {
        'features': [],
        'labels': [],
        'mapping': [],
        'config': {
            'latent_dim': latent_dim,
            'sample_rate': config.sample_rate,
            'song_length': song_length,
            'num_songs': 0,
            'extraction_type': 'cnn_autoencoder'
        }
    }
    
    # Load checkpoint if resuming
    start_idx = 0
    if resume and checkpoint_file and os.path.exists(checkpoint_file):
        logger.info(f"Resuming from checkpoint: {checkpoint_file}")
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            start_idx = checkpoint['processed_songs']
            output_data = checkpoint['output_data']
        logger.info(f"Resuming from song {start_idx + 1}")
    else:
        # Create empty output file
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Created empty output file: {output_file}")
    
    # Process songs with fresh autoencoders
    for song_idx in range(start_idx, len(audio_files)):
        file_path, genre = audio_files[song_idx]
        song_name = os.path.basename(file_path)
        
        logger.info(f"Processing song {song_idx + 1}/{len(audio_files)}: {song_name} ({genre})")
        
        # Create fresh autoencoder for this song
        fresh_autoencoder = SongLevelAutoencoder(
            song_length=int(config.sample_rate * song_length),
            latent_dim=latent_dim
        )
        fresh_autoencoder = fresh_autoencoder.to(extractor.device)
        
        try:
            # Load and preprocess song
            audio, sr = librosa.load(file_path, sr=config.sample_rate, duration=song_length)
            if len(audio) < config.sample_rate * song_length * 0.5:
                logger.warning(f"Skipping {file_path}: too short")
                continue
                
            # Normalize
            audio = (audio - audio.mean()) / (audio.std() + 1e-8)
            
            # Convert to tensor and add batch dimension
            audio_tensor = torch.FloatTensor(audio).unsqueeze(0).unsqueeze(0)  # (1, 1, length)
            audio_tensor = audio_tensor.to(extractor.device)
            
            # Train this fresh autoencoder with better settings for CNN
            criterion = nn.MSELoss()
            optimizer = optim.Adam(fresh_autoencoder.parameters(), lr=5e-4, weight_decay=1e-7)  # Much higher LR, minimal regularization
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=15)
            
            song_losses = []
            best_loss = float('inf')
            patience = 50  # Maximum patience for CNN
            patience_counter = 0
            improvement_threshold = 0.0001  # Very sensitive improvement threshold
            
            for epoch in range(epochs):
                fresh_autoencoder.train()
                optimizer.zero_grad()
                
                reconstructed, latent = fresh_autoencoder(audio_tensor)
                loss = criterion(reconstructed, audio_tensor)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fresh_autoencoder.parameters(), max_norm=1.0)  # Gradient clipping
                optimizer.step()
                
                current_loss = loss.item()
                song_losses.append(current_loss)
                
                # Update learning rate
                scheduler.step(current_loss)
                
                # Plateau breaking: if stuck for too long, increase learning rate temporarily
                if patience_counter == patience // 2:  # Halfway through patience
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 1.5  # Boost learning rate
                    logger.info(f"Song {song_idx + 1}: Boosting learning rate to {optimizer.param_groups[0]['lr']:.2e}")
                
                # Early stopping with improvement threshold (same as main training)
                if current_loss < best_loss - improvement_threshold:
                    best_loss = current_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    logger.info(f"Song {song_idx + 1}, Early stopping at epoch {epoch + 1} (no improvement > {improvement_threshold} for {patience} epochs)")
                    break
                
                if epoch % 5 == 0:
                    logger.info(f"Song {song_idx + 1}, Epoch {epoch + 1}/{epochs}, Loss: {current_loss:.6f}, Best: {best_loss:.6f}")
            
            # Extract final encoding
            with torch.no_grad():
                fresh_autoencoder.eval()
                _, final_encoding = fresh_autoencoder(audio_tensor)
                final_encoding = final_encoding.cpu().numpy().flatten()  # (latent_dim,)
            
            # Store results immediately
            # Format: features should be array of songs, each song is array of frames
            # For autoencoder: each song has 1 frame with latent_dim features
            song_features = [final_encoding.tolist()]  # Wrap in array to represent 1 frame
            
            # Add to output data
            output_data['features'].append(song_features)
            output_data['labels'].append(genre)
            if genre not in output_data['mapping']:
                output_data['mapping'].append(genre)
            output_data['config']['num_songs'] = len(output_data['features'])
            
            # Save immediately to file
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            logger.info(f"Song {song_idx + 1} completed. Encoding shape: {final_encoding.shape}")
            logger.info(f"Saved to {output_file} - {len(output_data['features'])} songs processed")
            
            # Save training plots for the first song only
            if song_idx == 0:
                output_dir = os.path.dirname(output_file)
                plot_file, detailed_plot_file = save_training_plots(
                    song_losses, output_dir, song_name, "cnn"
                )
                logger.info(f"Training plots saved: {plot_file}, {detailed_plot_file}")
            
            # Clean up
            del fresh_autoencoder
            torch.cuda.empty_cache() if extractor.device.type == 'cuda' else None
            
            # Save checkpoint every 10 songs
            if (song_idx + 1) % 10 == 0:
                checkpoint_data = {
                    'processed_songs': song_idx + 1,
                    'output_data': output_data
                }
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
                logger.info(f"Checkpoint saved: {len(output_data['features'])} songs processed")
            
        except Exception as e:
            logger.warning(f"Failed to process {file_path}: {e}")
            continue
    
    # Final processing: encode labels and sort mapping to match MFCC format
    unique_genres = sorted(output_data['mapping'])
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(output_data['labels'])
    
    # Update with encoded labels (integers 0-9)
    output_data['labels'] = encoded_labels.tolist()
    output_data['mapping'] = unique_genres
    
    # Remove config section to match MFCC format exactly
    if 'config' in output_data:
        del output_data['config']
    
    # Save final version with encoded labels
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"CNN autoencoded features saved to {output_file}")
    logger.info(f"Features shape: {len(output_data['features'])} x 1 x {latent_dim}")
    logger.info(f"Genres: {unique_genres}")
    
    # Clean up checkpoint file
    if checkpoint_file and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        logger.info("Checkpoint file cleaned up")
    
    return output_file


def extract_recurrent_features(gtzan_path: str, output_file: str, 
                              latent_dim: int = 64, hidden_size: int = 256,
                              epochs: int = 1000, song_length: float = 30.0,
                              checkpoint_file: str = None, resume: bool = False) -> str:
    """Extract RNN-based autoencoded features from GTZAN dataset."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Extracting RNN-based autoencoded features from {gtzan_path}")
    logger.info(f"Latent dimension: {latent_dim}, Hidden size: {hidden_size}, Max epochs: {epochs}")
    
    # Create audio config
    config = AudioConfig()
    config.sample_rate = SAMPLE_RATE
    config.max_duration = song_length
    
    # Get GTZAN audio files
    audio_files = []
    for genre in os.listdir(gtzan_path):
        genre_path = os.path.join(gtzan_path, genre)
        if os.path.isdir(genre_path):
            for file in os.listdir(genre_path):
                if file.endswith('.wav'):
                    audio_files.append((os.path.join(genre_path, file), genre))
    
    logger.info(f"Found {len(audio_files)} songs")
    
    # Initialize extractor
    extractor = RecurrentAutoencoderExtractor(
        config=config,
        latent_dim=latent_dim,
        song_length=song_length,
        hidden_size=hidden_size,
        logger=logger
    )
    
    # Initialize output file with empty structure
    output_data = {
        'features': [],
        'labels': [],
        'mapping': [],
        'config': {
            'latent_dim': latent_dim,
            'hidden_size': hidden_size,
            'sample_rate': config.sample_rate,
            'song_length': song_length,
            'num_songs': 0,
            'extraction_type': 'rnn_autoencoder'
        }
    }
    
    # Load checkpoint if resuming
    start_idx = 0
    if resume and checkpoint_file and os.path.exists(checkpoint_file):
        logger.info(f"Resuming from checkpoint: {checkpoint_file}")
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            start_idx = checkpoint['processed_songs']
            output_data = checkpoint['output_data']
        logger.info(f"Resuming from song {start_idx + 1}")
    else:
        # Create empty output file
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Created empty output file: {output_file}")
    
    # Process songs with fresh recurrent autoencoders
    for song_idx in range(start_idx, len(audio_files)):
        file_path, genre = audio_files[song_idx]
        song_name = os.path.basename(file_path)
        
        logger.info(f"Processing song {song_idx + 1}/{len(audio_files)}: {song_name} ({genre})")
        
        # Create fresh recurrent autoencoder for this song
        fresh_autoencoder = RecurrentAutoencoder(
            song_length=int(config.sample_rate * song_length),
            latent_dim=latent_dim,
            hidden_size=hidden_size
        )
        fresh_autoencoder = fresh_autoencoder.to(extractor.device)
        
        try:
            # Load and preprocess song
            audio, sr = librosa.load(file_path, sr=config.sample_rate, duration=song_length)
            if len(audio) < config.sample_rate * song_length * 0.5:
                logger.warning(f"Skipping {file_path}: too short")
                continue
                
            # Normalize
            audio = (audio - audio.mean()) / (audio.std() + 1e-8)
            
            # Convert to tensor
            audio_tensor = torch.FloatTensor(audio).unsqueeze(0)  # (1, length)
            audio_tensor = audio_tensor.to(extractor.device)
            
            # Train this fresh autoencoder with better settings for RNN
            criterion = nn.MSELoss()
            optimizer = optim.Adam(fresh_autoencoder.parameters(), lr=1e-3, weight_decay=1e-5)  # Much lower LR, less regularization
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=8)
            
            song_losses = []
            best_loss = float('inf')
            patience = 30  # Much more patience for RNN
            patience_counter = 0
            improvement_threshold = 0.0005  # More sensitive improvement threshold
            
            for epoch in range(epochs):
                fresh_autoencoder.train()
                optimizer.zero_grad()
                
                reconstructed, latent = fresh_autoencoder(audio_tensor)
                loss = criterion(reconstructed, audio_tensor)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fresh_autoencoder.parameters(), max_norm=1.0)  # Gradient clipping
                optimizer.step()
                
                current_loss = loss.item()
                song_losses.append(current_loss)
                
                # Update learning rate
                scheduler.step(current_loss)
                
                # Plateau breaking: if stuck for too long, increase learning rate temporarily
                if patience_counter == patience // 2:  # Halfway through patience
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 1.5  # Boost learning rate
                    logger.info(f"Song {song_idx + 1}: Boosting learning rate to {optimizer.param_groups[0]['lr']:.2e}")
                
                # Early stopping with improvement threshold (same as main training)
                if current_loss < best_loss - improvement_threshold:
                    best_loss = current_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    logger.info(f"Song {song_idx + 1}, Early stopping at epoch {epoch + 1} (no improvement > {improvement_threshold} for {patience} epochs)")
                    break
                
                if epoch % 5 == 0:
                    logger.info(f"Song {song_idx + 1}, Epoch {epoch + 1}/{epochs}, Loss: {current_loss:.6f}, Best: {best_loss:.6f}")
            
            # Extract final encoding
            with torch.no_grad():
                fresh_autoencoder.eval()
                _, final_encoding = fresh_autoencoder(audio_tensor)
                final_encoding = final_encoding.cpu().numpy().flatten()  # (latent_dim,)
            
            # Store results immediately
            # Format: features should be array of songs, each song is array of frames
            # For autoencoder: each song has 1 frame with latent_dim features
            song_features = [final_encoding.tolist()]  # Wrap in array to represent 1 frame
            
            # Add to output data
            output_data['features'].append(song_features)
            output_data['labels'].append(genre)
            if genre not in output_data['mapping']:
                output_data['mapping'].append(genre)
            output_data['config']['num_songs'] = len(output_data['features'])
            
            # Save immediately to file
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            logger.info(f"Song {song_idx + 1} completed. Encoding shape: {final_encoding.shape}")
            logger.info(f"Saved to {output_file} - {len(output_data['features'])} songs processed")
            
            # Save training plots for the first song only
            if song_idx == 0:
                output_dir = os.path.dirname(output_file)
                plot_file, detailed_plot_file = save_training_plots(
                    song_losses, output_dir, song_name, "rnn"
                )
                logger.info(f"Training plots saved: {plot_file}, {detailed_plot_file}")
            
            # Clean up
            del fresh_autoencoder
            torch.cuda.empty_cache() if extractor.device.type == 'cuda' else None
            
            # Save checkpoint every 10 songs
            if (song_idx + 1) % 10 == 0:
                checkpoint_data = {
                    'processed_songs': song_idx + 1,
                    'output_data': output_data
                }
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
                logger.info(f"Checkpoint saved: {len(output_data['features'])} songs processed")
            
        except Exception as e:
            logger.warning(f"Failed to process {file_path}: {e}")
            continue
    
    # Final processing: encode labels and sort mapping to match MFCC format
    unique_genres = sorted(output_data['mapping'])
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(output_data['labels'])
    
    # Update with encoded labels (integers 0-9)
    output_data['labels'] = encoded_labels.tolist()
    output_data['mapping'] = unique_genres
    
    # Remove config section to match MFCC format exactly
    if 'config' in output_data:
        del output_data['config']
    
    # Save final version with encoded labels
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"RNN autoencoded features saved to {output_file}")
    logger.info(f"Features shape: {len(output_data['features'])} x 1 x {latent_dim}")
    logger.info(f"Genres: {unique_genres}")
    
    # Clean up checkpoint file
    if checkpoint_file and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        logger.info("Checkpoint file cleaned up")
    
    return output_file


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Extract features from GTZAN dataset")
    parser.add_argument("--gtzan-path", required=True, help="Path to GTZAN dataset directory")
    parser.add_argument("--output-file", required=True, help="Output JSON file path")
    parser.add_argument("--approach", choices=["mfcc", "cnn_autoencoder", "rnn_autoencoder"], 
                       default="mfcc", help="Feature extraction approach")
    
    # MFCC parameters
    parser.add_argument("--n-mfcc", type=int, default=13, help="Number of MFCC coefficients")
    parser.add_argument("--hop-length", type=int, default=512, help="Hop length for MFCC")
    
    # Autoencoder parameters
    parser.add_argument("--latent-dim", type=int, default=64, help="Latent dimension for autoencoders")
    parser.add_argument("--hidden-size", type=int, default=256, help="LSTM hidden size (for RNN)")
    parser.add_argument("--epochs", type=int, default=1000, help="Max epochs per song (early stopping)")
    parser.add_argument("--song-length", type=float, default=30.0, help="Song length in seconds")
    
    # Checkpoint parameters
    parser.add_argument("--checkpoint-file", help="Checkpoint file for resuming")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    print("=" * 60)
    print("🎵 GTZAN FEATURE EXTRACTION")
    print("=" * 60)
    print(f"Approach: {args.approach.upper()}")
    print(f"Output: {args.output_file}")
    print("=" * 60)
    
    if args.approach == "mfcc":
        print("Features:")
        print("✅ Traditional MFCC features")
        print(f"✅ {args.n_mfcc} MFCC coefficients")
        print(f"✅ Hop length: {args.hop_length}")
        print("✅ Fast extraction")
        
        extract_mfcc_features(
            gtzan_path=args.gtzan_path,
            output_file=args.output_file,
            n_mfcc=args.n_mfcc,
            hop_length=args.hop_length,
            song_length=args.song_length
        )
        
    elif args.approach == "cnn_autoencoder":
        print("Features:")
        print("✅ CNN-based autoencoder")
        print("✅ Captures spatial patterns")
        print("✅ Early stopping (no improvement > 0.01% for 10 epochs)")
        print("✅ One autoencoder per song")
        print(f"✅ {args.latent_dim}D latent space")
        
        extract_autoencoded_features(
            gtzan_path=args.gtzan_path,
            output_file=args.output_file,
            latent_dim=args.latent_dim,
            epochs=args.epochs,
            song_length=args.song_length,
            checkpoint_file=args.checkpoint_file,
            resume=args.resume
        )
        
    elif args.approach == "rnn_autoencoder":
        print("Features:")
        print("✅ RNN-based autoencoder (LSTM)")
        print("✅ Captures temporal patterns")
        print("✅ Early stopping (no improvement > 0.01% for 10 epochs)")
        print("✅ One autoencoder per song")
        print(f"✅ {args.latent_dim}D latent space")
        print(f"✅ {args.hidden_size} LSTM hidden size")
        
        extract_recurrent_features(
            gtzan_path=args.gtzan_path,
            output_file=args.output_file,
            latent_dim=args.latent_dim,
            hidden_size=args.hidden_size,
            epochs=args.epochs,
            song_length=args.song_length,
            checkpoint_file=args.checkpoint_file,
            resume=args.resume
        )
    
    print("\n✅ Feature extraction complete!")
    print(f"Output file: {args.output_file}")
    print("\nExample usage:")
    print(f"python src/training/train_model.py --data {args.output_file} --model FC --output outputs/fc_{args.approach}")


if __name__ == "__main__":
    main()
