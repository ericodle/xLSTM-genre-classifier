#!/usr/bin/env python3
"""
Song-Level Autoencoder for Music Genre Classification

This module implements an autoencoder that processes entire songs as single units,
encoding each song into a single latent vector that captures the full musical structure.

Key differences from segment-based approach:
- Input: Full songs (30 seconds, 661,500 samples)
- Output: Single latent vector per song (e.g., 256 dimensions)
- Captures: Long-term musical patterns, song structure, genre characteristics
"""

import os
import sys
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.config import AudioConfig
from core.constants import SAMPLE_RATE


class SongLevelDataset(Dataset):
    """Dataset for loading full songs as single units."""
    
    def __init__(self, audio_files: List[Tuple[str, str]], sample_rate: int = 22050, 
                 song_length: float = 30.0, normalize: bool = True):
        """
        Initialize song-level dataset.
        
        Args:
            audio_files: List of (file_path, genre) tuples
            sample_rate: Target sample rate for audio
            song_length: Length of songs in seconds (will pad/truncate)
            normalize: Whether to normalize audio
        """
        self.audio_files = audio_files
        self.sample_rate = sample_rate
        self.song_length = song_length
        self.normalize = normalize
        self.song_samples = int(sample_rate * song_length)
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        file_path, genre = self.audio_files[idx]
        
        try:
            # Load full song
            try:
                audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.song_length)
            except Exception as e:
                logging.warning(f"Failed to load {file_path}: {e}")
                # Return silence if loading fails
                audio = np.zeros(self.song_samples)
            
            # Ensure correct length
            if len(audio) < self.song_samples:
                audio = np.pad(audio, (0, self.song_samples - len(audio)), mode='constant')
            else:
                audio = audio[:self.song_samples]
            
            # Normalize audio
            if self.normalize and np.max(np.abs(audio)) > 0:
                audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
            return torch.FloatTensor(audio), genre
            
        except Exception as e:
            logging.warning(f"Failed to process {file_path}: {e}")
            # Return silence if processing fails
            return torch.zeros(self.song_samples), "unknown"


class SongLevelAutoencoder(nn.Module):
    """Autoencoder for full songs using 1D convolutions."""
    
    def __init__(self, song_length: int = 661500,  # 30s at 22050Hz
                 latent_dim: int = 256):
        """
        Initialize song-level autoencoder.
        
        Args:
            song_length: Length of input songs in samples
            latent_dim: Dimension of latent representation
        """
        super().__init__()
        
        self.song_length = song_length
        self.latent_dim = latent_dim
        
        # Encoder: Compress full song to latent vector
        self.encoder = nn.Sequential(
            # Input: (batch, 1, 661500)
            nn.Conv1d(1, 64, kernel_size=31, stride=4, padding=15),  # -> (batch, 64, 165376)
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=31, stride=4, padding=15), # -> (batch, 128, 41344)
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, kernel_size=31, stride=4, padding=15), # -> (batch, 256, 10336)
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 512, kernel_size=31, stride=4, padding=15), # -> (batch, 512, 2584)
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.AdaptiveAvgPool1d(1),  # -> (batch, 512, 1)
            nn.Flatten(),  # -> (batch, 512)
            nn.Linear(512, latent_dim),  # -> (batch, latent_dim)
            nn.ReLU()
        )
        
        # Decoder: Reconstruct song from latent vector
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),  # -> (batch, 512)
            nn.ReLU(),
            nn.Unflatten(1, (512, 1)),  # -> (batch, 512, 1)
            nn.ConvTranspose1d(512, 256, kernel_size=31, stride=4, padding=15, output_padding=3),  # -> (batch, 256, 7)
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.ConvTranspose1d(256, 128, kernel_size=31, stride=4, padding=15, output_padding=3),  # -> (batch, 128, 31)
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.ConvTranspose1d(128, 64, kernel_size=31, stride=4, padding=15, output_padding=3),   # -> (batch, 64, 127)
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.ConvTranspose1d(64, 1, kernel_size=31, stride=4, padding=15, output_padding=3),     # -> (batch, 1, 511)
            nn.AdaptiveAvgPool1d(song_length),  # -> (batch, 1, song_length)
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode full song to latent representation."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to full song."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: encode and decode."""
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z


class SongLevelAutoencoderExtractor:
    """Song-level autoencoder feature extractor."""
    
    def __init__(self, config: AudioConfig, latent_dim: int = 256, 
                 song_length: float = 30.0, device: str = "auto", 
                 logger: Optional[logging.Logger] = None):
        """
        Initialize song-level autoencoder extractor.
        
        Args:
            config: Audio configuration
            latent_dim: Dimension of latent representation
            song_length: Length of songs in seconds
            device: Device to use for training
            logger: Logger instance
        """
        self.config = config
        self.latent_dim = latent_dim
        self.song_length = song_length
        self.logger = logger or logging.getLogger(__name__)
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize model
        song_samples = int(config.sample_rate * song_length)
        self.model = SongLevelAutoencoder(
            song_length=song_samples,
            latent_dim=latent_dim
        ).to(self.device)
        
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
    def train_autoencoder(self, audio_files: List[Tuple[str, str]], 
                         batch_size: int = 8, epochs: int = 30,
                         learning_rate: float = 1e-3, validation_split: float = 0.2,
                         save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Train the song-level autoencoder on full songs.
        """
        self.logger.info(f"Training song-level autoencoder on {len(audio_files)} songs")
        self.logger.info(f"Song length: {self.song_length}s, Latent dim: {self.latent_dim}")
        
        # Create dataset
        dataset = SongLevelDataset(
            audio_files=audio_files,
            sample_rate=self.config.sample_rate,
            song_length=self.song_length,
            normalize=True
        )
        
        # Split dataset
        train_size = int((1 - validation_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        # Setup training
        self.model = self.model.to(self.device)  # Move model to device
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, (audio, _) in enumerate(train_loader):
                audio = audio.to(self.device)
                
                # Add channel dimension for conv1d
                if len(audio.shape) == 2:
                    audio = audio.unsqueeze(1)  # (batch, 1, length)
                
                optimizer.zero_grad()
                
                # Forward pass
                reconstructed, _ = self.model(audio)
                
                # Compute loss
                loss = criterion(reconstructed, audio)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                if batch_idx % 5 == 0:  # Log less frequently due to longer processing
                    self.logger.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for audio, _ in val_loader:
                    audio = audio.to(self.device)
                    
                    if len(audio.shape) == 2:
                        audio = audio.unsqueeze(1)
                    
                    reconstructed, _ = self.model(audio)
                    loss = criterion(reconstructed, audio)
                    val_loss += loss.item()
            
            # Calculate average losses
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            # Update learning rate
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Store history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['learning_rates'].append(current_lr)
            
            # Log progress
            self.logger.info(f'Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.6f}')
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if save_path:
                    self.save_model(save_path)
                    self.logger.info(f'New best model saved with val_loss: {best_val_loss:.6f}')
        
        self.is_trained = True
        self.logger.info("Song-level autoencoder training completed!")
        
        return history
    
    def train_autoencoder_per_song(self, audio_files: List[Tuple[str, str]], 
                                  epochs_per_song: int = 10, learning_rate: float = 1e-3,
                                  save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Train a FRESH autoencoder on each song individually for specified epochs.
        Each song gets its own completely independent autoencoder model.
        This approach ensures no cross-contamination between songs.
        """
        self.logger.info(f"Training FRESH song-level autoencoders per-song on {len(audio_files)} songs")
        self.logger.info(f"Song length: {self.song_length}s, Latent dim: {self.latent_dim}")
        self.logger.info(f"Epochs per song: {epochs_per_song}")
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'song_losses': [],
            'song_encodings': []  # Store the final encodings
        }
        
        # Process each song with a fresh autoencoder
        for song_idx, (file_path, genre) in enumerate(audio_files):
            self.logger.info(f"Training FRESH autoencoder on song {song_idx+1}/{len(audio_files)}: {os.path.basename(file_path)} ({genre})")
            
            # Create a FRESH autoencoder for this song
            fresh_autoencoder = SongLevelAutoencoder(
                song_length=int(self.config.sample_rate * self.song_length),
                latent_dim=self.latent_dim
            )
            fresh_autoencoder = fresh_autoencoder.to(self.device)
            
            # Load and preprocess single song
            try:
                audio, sr = librosa.load(file_path, sr=self.config.sample_rate, duration=self.song_length)
                if len(audio) < self.config.sample_rate * self.song_length * 0.5:  # Skip if too short
                    self.logger.warning(f"Skipping {file_path}: too short")
                    continue
                    
                # Normalize
                audio = (audio - audio.mean()) / (audio.std() + 1e-8)
                
                # Convert to tensor and add batch dimension
                audio_tensor = torch.FloatTensor(audio).unsqueeze(0).unsqueeze(0)  # (1, 1, length)
                audio_tensor = audio_tensor.to(self.device)
                
                # Setup training for this fresh model
                criterion = nn.MSELoss()
                optimizer = optim.Adam(fresh_autoencoder.parameters(), lr=learning_rate)
                
                # Train this fresh autoencoder on this song only
                song_losses = []
                for epoch in range(epochs_per_song):
                    fresh_autoencoder.train()
                    optimizer.zero_grad()
                    
                    # Forward pass
                    reconstructed, latent = fresh_autoencoder(audio_tensor)
                    
                    # Compute loss
                    loss = criterion(reconstructed, audio_tensor)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    song_losses.append(loss.item())
                    
                    if epoch % 2 == 0:  # Log every 2 epochs
                        self.logger.info(f"Song {song_idx+1}, Epoch {epoch+1}/{epochs_per_song}, Loss: {loss.item():.6f}")
                
                # Extract final encoding from this trained model
                with torch.no_grad():
                    fresh_autoencoder.eval()
                    _, final_encoding = fresh_autoencoder(audio_tensor)
                    final_encoding = final_encoding.cpu().numpy().flatten()  # (latent_dim,)
                
                # Store results
                avg_song_loss = np.mean(song_losses)
                history['train_loss'].append(avg_song_loss)
                history['val_loss'].append(avg_song_loss)  # Same as train since no validation split
                history['learning_rates'].append(optimizer.param_groups[0]['lr'])
                history['song_losses'].append(song_losses)
                history['song_encodings'].append(final_encoding)
                
                self.logger.info(f"Song {song_idx+1} completed. Avg Loss: {avg_song_loss:.6f}, Encoding shape: {final_encoding.shape}")
                
                # Clean up this model to free memory
                del fresh_autoencoder
                torch.cuda.empty_cache() if self.device.type == 'cuda' else None
                
            except Exception as e:
                self.logger.warning(f"Failed to process {file_path}: {e}")
                continue
        
        self.logger.info("FRESH per-song autoencoder training completed!")
        self.logger.info(f"Generated {len(history['song_encodings'])} song encodings with shape {history['song_encodings'][0].shape if history['song_encodings'] else 'None'}")
        
        return history
    
    def extract_song_encodings(self, audio_files: List[Tuple[str, str]], 
                              batch_size: int = 8) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract song-level encodings using the trained autoencoder.
        Each song becomes a single latent vector.
        """
        if not self.is_trained:
            raise ValueError("Autoencoder must be trained before extracting encodings")
        
        self.logger.info(f"Extracting song encodings from {len(audio_files)} songs")
        
        # Create dataset
        dataset = SongLevelDataset(
            audio_files=audio_files,
            sample_rate=self.config.sample_rate,
            song_length=self.song_length,
            normalize=True
        )
        
        # Create data loader
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        # Extract encodings
        encodings = []
        labels = []
        
        self.model.eval()
        with torch.no_grad():
            for audio, genre in data_loader:
                audio = audio.to(self.device)
                
                if len(audio.shape) == 2:
                    audio = audio.unsqueeze(1)
                
                # Get song-level encoding (one vector per song)
                song_encoding = self.model.encode(audio)
                encodings.append(song_encoding.cpu().numpy())
                labels.extend(genre)
        
        # Concatenate encodings
        encodings = np.vstack(encodings)
        
        # Encode labels
        unique_genres = sorted(list(set(labels)))
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        self.logger.info(f"Extracted song encodings shape: {encodings.shape}")
        self.logger.info(f"Number of unique genres: {len(unique_genres)}")
        
        return encodings, encoded_labels, unique_genres
    
    def save_model(self, path: str) -> None:
        """Save the trained autoencoder model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'latent_dim': self.latent_dim,
            'song_length': self.song_length,
            'label_encoder': self.label_encoder,
            'is_trained': self.is_trained
        }, path)
        
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load a trained autoencoder model."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.label_encoder = checkpoint['label_encoder']
        self.is_trained = checkpoint['is_trained']
        
        self.logger.info(f"Model loaded from {path}")


def extract_song_level_features(gtzan_path: str, output_path: str, 
                               latent_dim: int = 256, epochs: int = 30,
                               batch_size: int = 8, song_length: float = 30.0) -> str:
    """
    Extract song-level features from GTZAN dataset using song-level autoencoder.
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
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
    
    # Train autoencoder
    logger.info("Training song-level autoencoder...")
    history = extractor.train_autoencoder(
        audio_files=audio_files,
        batch_size=batch_size,
        epochs=epochs,
        save_path=os.path.join(output_path, "song_level_autoencoder_model.pth")
    )
    
    # Extract song encodings
    logger.info("Extracting song encodings...")
    encodings, labels, genres = extractor.extract_song_encodings(
        audio_files=audio_files,
        batch_size=batch_size
    )
    
    # Save features
    output_data = {
        'song_encodings': encodings.tolist(),
        'labels': labels.tolist(),
        'mapping': genres,
        'config': {
            'latent_dim': latent_dim,
            'sample_rate': config.sample_rate,
            'song_length': song_length,
            'num_songs': len(audio_files)
        }
    }
    
    features_file = os.path.join(output_path, "song_level_features.json")
    with open(features_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Song encodings saved to {features_file}")
    logger.info(f"Encodings shape: {encodings.shape}")
    logger.info(f"Genres: {genres}")
    
    return features_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract song-level features using autoencoder")
    parser.add_argument("--gtzan-path", required=True, help="Path to GTZAN dataset")
    parser.add_argument("--output-path", required=True, help="Output directory")
    parser.add_argument("--latent-dim", type=int, default=256, help="Latent dimension")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--song-length", type=float, default=30.0, help="Song length in seconds")
    
    args = parser.parse_args()
    
    extract_song_level_features(
        gtzan_path=args.gtzan_path,
        output_path=args.output_path,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        song_length=args.song_length
    )
