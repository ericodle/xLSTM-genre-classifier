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
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract autoencoded features for GTZAN dataset")
    parser.add_argument("--latent_dim", type=int, default=64, help="Latent dimension for autoencoder (default: 64)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory (default: outputs)")
    
    args = parser.parse_args()
    
    gtzan_path = "/home/eo/Documents/gtzan"
    song_output = f"{args.output_dir}/gtzan_song_level_autoencoded_{args.latent_dim}d.json"
    checkpoint_file = f"{args.output_dir}/gtzan_song_level_autoencoded_{args.latent_dim}d_checkpoint.json"
    
    print("🎵 Extracting Fresh Autoencoded Features for GTZAN Dataset")
    print("=" * 60)
    print("Features:")
    print("✅ Incremental saving (every 10 songs)")
    print("✅ Checkpoint resuming")
    print("✅ One autoencoder per song")
    print(f"✅ {args.latent_dim}D latent space")
    print("✅ 50 epochs per song")
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
        latent_dim=args.latent_dim,
        epochs=50,  # Hard-coded to 50 epochs
        song_length=30.0,
        checkpoint_file=checkpoint_file,
        resume=resume
    )
    
    print("\n✅ Fresh autoencoder extraction complete!")
    print(f"Song-level features: {song_output}")
    print("\nThis file can now be used with your existing training framework!")


if __name__ == "__main__":
    main()
