#!/usr/bin/env python3
"""
Recurrent Autoencoder for Music Genre Classification

This module implements an RNN-based autoencoder that processes music as sequential data,
capturing temporal patterns and musical structure through recurrence.

Key features:
- Input: Full songs (30 seconds, 661,500 samples)
- Architecture: LSTM-based encoder-decoder
- Output: Single latent vector per song
- Captures: Long-term temporal patterns, musical motifs, genre characteristics
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


class RecurrentAutoencoder(nn.Module):
    """RNN-based autoencoder for music using LSTM encoder-decoder architecture."""
    
    def __init__(self, song_length: int = 661500,  # 30s at 22050Hz
                 latent_dim: int = 256, hidden_size: int = 512, 
                 num_layers: int = 2, dropout: float = 0.1):
        """
        Initialize recurrent autoencoder.
        
        Args:
            song_length: Length of input songs in samples
            latent_dim: Dimension of latent representation
            hidden_size: Hidden size of LSTM layers
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.song_length = song_length
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Downsample audio to manageable sequence length for RNN
        self.downsample_factor = 50  # Reduce 661500 to 13230 samples (less aggressive)
        self.downsampled_length = song_length // self.downsample_factor
        
        # Encoder: LSTM to compress temporal patterns
        self.encoder_lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Project to latent space
        self.latent_projection = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, latent_dim),
            nn.ReLU()
        )
        
        # Decoder: LSTM to reconstruct temporal patterns
        self.decoder_lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Project from latent space to RNN input
        self.latent_expansion = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size * 2)  # *2 for bidirectional
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size * 2, 1)
        
    def downsample_audio(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample audio to manageable sequence length."""
        # x: (batch, 1, song_length)
        # Use average pooling for downsampling
        x = x.view(x.size(0), 1, -1, self.downsample_factor)  # (batch, 1, downsampled_length, downsample_factor)
        x = x.mean(dim=3)  # (batch, 1, downsampled_length)
        return x.squeeze(1)  # (batch, downsampled_length)
    
    def upsample_audio(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample audio back to original length."""
        # x: (batch, downsampled_length)
        x = x.unsqueeze(1)  # (batch, 1, downsampled_length)
        x = x.repeat(1, 1, self.downsample_factor)  # (batch, 1, song_length)
        return x.squeeze(1)  # (batch, song_length)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode full song to latent representation."""
        # x: (batch, song_length) -> (batch, 1, song_length)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        # Downsample
        x_downsampled = self.downsample_audio(x)  # (batch, downsampled_length)
        
        # Add sequence dimension for LSTM
        x_seq = x_downsampled.unsqueeze(-1)  # (batch, downsampled_length, 1)
        
        # LSTM encoding
        lstm_out, (hidden, cell) = self.encoder_lstm(x_seq)
        
        # Use final hidden state from both directions
        # hidden: (num_layers * 2, batch, hidden_size)
        final_hidden = hidden[-1]  # (batch, hidden_size) - last layer
        final_hidden_reverse = hidden[-2]  # (batch, hidden_size) - reverse direction
        
        # Concatenate forward and backward
        combined_hidden = torch.cat([final_hidden, final_hidden_reverse], dim=1)  # (batch, hidden_size * 2)
        
        # Project to latent space
        latent = self.latent_projection(combined_hidden)  # (batch, latent_dim)
        
        return latent
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to full song."""
        batch_size = z.size(0)
        
        # Expand latent to RNN hidden state
        hidden_expanded = self.latent_expansion(z)  # (batch, hidden_size * 2)
        
        # Split into forward and backward components
        hidden_forward = hidden_expanded[:, :self.hidden_size]  # (batch, hidden_size)
        hidden_reverse = hidden_expanded[:, self.hidden_size:]  # (batch, hidden_size)
        
        # Create initial hidden states for decoder (accounting for num_layers)
        # LSTM expects (num_layers * num_directions, batch, hidden_size)
        h_0 = torch.stack([hidden_forward, hidden_reverse], dim=0)  # (2, batch, hidden_size)
        # Repeat for each layer: (2, batch, hidden_size) -> (num_layers, 2, batch, hidden_size)
        h_0 = h_0.unsqueeze(0).expand(self.num_layers, -1, -1, -1)  # (num_layers, 2, batch, hidden_size)
        h_0 = h_0.contiguous().view(self.num_layers * 2, batch_size, self.hidden_size)  # (num_layers * 2, batch, hidden_size)
        
        c_0 = torch.zeros_like(h_0)  # (num_layers * 2, batch, hidden_size)
        
        # Create input sequence (zeros initially)
        input_seq = torch.zeros(batch_size, self.downsampled_length, 1, device=z.device)
        
        # LSTM decoding
        lstm_out, _ = self.decoder_lstm(input_seq, (h_0, c_0))
        
        # Project to output
        output = self.output_projection(lstm_out)  # (batch, downsampled_length, 1)
        output = output.squeeze(-1)  # (batch, downsampled_length)
        
        # Upsample back to original length
        reconstructed = self.upsample_audio(output)  # (batch, song_length)
        
        return reconstructed
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: encode and decode."""
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z


class RecurrentAutoencoderExtractor:
    """Recurrent autoencoder feature extractor."""
    
    def __init__(self, config: AudioConfig, latent_dim: int = 256, 
                 song_length: float = 30.0, hidden_size: int = 512,
                 device: str = "auto", logger: Optional[logging.Logger] = None):
        """
        Initialize recurrent autoencoder extractor.
        
        Args:
            config: Audio configuration
            latent_dim: Dimension of latent representation
            song_length: Length of songs in seconds
            hidden_size: Hidden size of LSTM layers
            device: Device to use for training
            logger: Logger instance
        """
        self.config = config
        self.latent_dim = latent_dim
        self.song_length = song_length
        self.hidden_size = hidden_size
        self.logger = logger or logging.getLogger(__name__)
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize model
        song_samples = int(config.sample_rate * song_length)
        self.model = RecurrentAutoencoder(
            song_length=song_samples,
            latent_dim=latent_dim,
            hidden_size=hidden_size
        ).to(self.device)
        
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
    def train_autoencoder_per_song(self, audio_files: List[Tuple[str, str]], 
                                  epochs_per_song: int = 10, learning_rate: float = 1e-3,
                                  save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Train a FRESH recurrent autoencoder on each song individually.
        Each song gets its own completely independent autoencoder model.
        """
        self.logger.info(f"Training FRESH recurrent autoencoders per-song on {len(audio_files)} songs")
        self.logger.info(f"Song length: {self.song_length}s, Latent dim: {self.latent_dim}")
        self.logger.info(f"Hidden size: {self.hidden_size}, Epochs per song: {epochs_per_song}")
        
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
            self.logger.info(f"Training FRESH recurrent autoencoder on song {song_idx+1}/{len(audio_files)}: {os.path.basename(file_path)} ({genre})")
            
            # Create a FRESH autoencoder for this song
            fresh_autoencoder = RecurrentAutoencoder(
                song_length=int(self.config.sample_rate * self.song_length),
                latent_dim=self.latent_dim,
                hidden_size=self.hidden_size
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
                
                # Convert to tensor
                audio_tensor = torch.FloatTensor(audio).unsqueeze(0)  # (1, length)
                audio_tensor = audio_tensor.to(self.device)
                
                # Setup training for this fresh model
                criterion = nn.MSELoss()
                optimizer = optim.Adam(fresh_autoencoder.parameters(), lr=learning_rate, weight_decay=1e-5)
                
                # Training with early stopping
                song_losses = []
                best_loss = float('inf')
                patience = 10
                patience_counter = 0
                improvement_threshold = 0.0001  # 0.01% improvement
                
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
                    
                    current_loss = loss.item()
                    song_losses.append(current_loss)
                    
                    # Early stopping with improvement threshold
                    if current_loss < best_loss - improvement_threshold:
                        best_loss = current_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        self.logger.info(f"Song {song_idx + 1}, Early stopping at epoch {epoch + 1} (no improvement > {improvement_threshold} for {patience} epochs)")
                        break
                    
                    if epoch % 5 == 0:
                        self.logger.info(f"Song {song_idx+1}, Epoch {epoch+1}/{epochs_per_song}, Loss: {current_loss:.6f}, Best: {best_loss:.6f}")
                
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
        
        self.logger.info("FRESH per-song recurrent autoencoder training completed!")
        self.logger.info(f"Generated {len(history['song_encodings'])} song encodings with shape {history['song_encodings'][0].shape if history['song_encodings'] else 'None'}")
        
        return history


def extract_recurrent_features(gtzan_path: str, output_path: str, 
                              latent_dim: int = 256, hidden_size: int = 512,
                              epochs: int = 30, song_length: float = 30.0) -> str:
    """
    Extract recurrent features from GTZAN dataset using recurrent autoencoder.
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
    extractor = RecurrentAutoencoderExtractor(
        config=config,
        latent_dim=latent_dim,
        song_length=song_length,
        hidden_size=hidden_size,
        logger=logger
    )
    
    # Train autoencoders per song
    logger.info("Training recurrent autoencoders per song...")
    history = extractor.train_autoencoder_per_song(
        audio_files=audio_files,
        epochs_per_song=epochs
    )
    
    # Save features
    output_data = {
        'features': [encoding.tolist() for encoding in history['song_encodings']],
        'labels': [i % 10 for i in range(len(history['song_encodings']))],  # Placeholder labels
        'mapping': ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'],
        'config': {
            'latent_dim': latent_dim,
            'hidden_size': hidden_size,
            'sample_rate': config.sample_rate,
            'song_length': song_length,
            'num_songs': len(history['song_encodings'])
        }
    }
    
    features_file = os.path.join(output_path, "recurrent_features.json")
    with open(features_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Recurrent features saved to {features_file}")
    logger.info(f"Features shape: {len(history['song_encodings'])} x {latent_dim}")
    
    return features_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract recurrent features using RNN autoencoder")
    parser.add_argument("--gtzan-path", required=True, help="Path to GTZAN dataset")
    parser.add_argument("--output-path", required=True, help="Output directory")
    parser.add_argument("--latent-dim", type=int, default=256, help="Latent dimension")
    parser.add_argument("--hidden-size", type=int, default=512, help="LSTM hidden size")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs per song")
    parser.add_argument("--song-length", type=float, default=30.0, help="Song length in seconds")
    
    args = parser.parse_args()
    
    extract_recurrent_features(
        gtzan_path=args.gtzan_path,
        output_path=args.output_path,
        latent_dim=args.latent_dim,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        song_length=args.song_length
    )
