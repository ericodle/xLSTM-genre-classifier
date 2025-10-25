#!/usr/bin/env python3
"""
Extract autoencoded features for GTZAN dataset in MFCC-compatible format.
"""

import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.extract_autoencoded_features import extract_song_level_features_json


def main():
    """Extract fresh song-level autoencoded features with checkpoint support."""
    
    gtzan_path = "/home/eo/Documents/gtzan"
    song_output = "outputs/gtzan_song_level_autoencoded.json"
    checkpoint_file = "outputs/gtzan_song_level_autoencoded_checkpoint.json"
    
    print("🎵 Extracting Fresh Autoencoded Features for GTZAN Dataset")
    print("=" * 60)
    print("Features:")
    print("✅ Incremental saving (every 10 songs)")
    print("✅ Checkpoint resuming")
    print("✅ One autoencoder per song")
    print("✅ 128D latent space")
    print("=" * 60)
    
    # Check if resuming
    import os
    if os.path.exists(checkpoint_file):
        print(f"\n📂 Checkpoint found: {checkpoint_file}")
        print("🔄 Automatically resuming from checkpoint...")
        resume = True
    else:
        resume = False
    
    # Fresh song-level features (30s songs, 128D latent space)
    print(f"\n{'Resuming' if resume else 'Starting'} Fresh Song-Level Autoencoded Features")
    print("-" * 50)
    
    extract_song_level_features_json(
        gtzan_path=gtzan_path,
        output_file=song_output,
        latent_dim=128,  # 128D latent space
        epochs=10,       # 10 epochs per song
        song_length=30.0,
        checkpoint_file=checkpoint_file,
        resume=resume
    )
    
    print("\n✅ Fresh autoencoder extraction complete!")
    print(f"Song-level features: {song_output}")
    print("\nThis file can now be used with your existing training framework!")


if __name__ == "__main__":
    main()
