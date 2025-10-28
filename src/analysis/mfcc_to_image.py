#!/usr/bin/env python3
"""
Convert MFCC data to 2D images for visualization and analysis.

This script loads MFCC JSON data and converts each sample to a 2D image
where the x-axis represents time and y-axis represents MFCC coefficients.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_mfcc_data(json_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load MFCC data from JSON file."""
    logger.info(f"Loading MFCC data from {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    if "features" in data and "labels" in data:
        # New format
        features_list = data["features"]
        labels = np.array(data["labels"])

        # Get genre mapping
        if "mapping" in data:
            genre_names = data["mapping"]
        else:
            unique_labels = sorted(list(set(labels)))
            genre_names = [f"Genre_{i}" for i in unique_labels]

        # Pad features to consistent length
        max_length = max(len(f) for f in features_list)
        logger.info(f"Padding features to max length: {max_length}")

        padded_features = []
        for feature in features_list:
            if len(feature) < max_length:
                padding_needed = max_length - len(feature)
                padded_feature = feature + [[0.0] * len(feature[0]) for _ in range(padding_needed)]
                padded_features.append(padded_feature)
            else:
                padded_features.append(feature)

        features = np.array(padded_features, dtype=np.float32)

        # Convert string labels to integers if needed
        if labels.dtype == object:
            label_to_idx = {label: idx for idx, label in enumerate(genre_names)}
            labels = np.array([label_to_idx[label] for label in labels])

        logger.info(f"Loaded {len(features)} samples with shape {features.shape}")
        return features, labels, genre_names

    else:
        raise ValueError("Unsupported JSON format. Expected 'features' and 'labels' keys.")


def normalize_mfcc_for_image(mfcc_data: np.ndarray) -> np.ndarray:
    """Normalize MFCC data for image visualization."""
    # Normalize to 0-1 range
    mfcc_min = np.min(mfcc_data)
    mfcc_max = np.max(mfcc_data)

    if mfcc_max > mfcc_min:
        normalized = (mfcc_data - mfcc_min) / (mfcc_max - mfcc_min)
    else:
        normalized = np.zeros_like(mfcc_data)

    return normalized


def create_mfcc_image(
    mfcc_data: np.ndarray, title: str = "MFCC Spectrogram", figsize: Tuple[int, int] = (12, 4)
) -> plt.Figure:
    """Create a 2D image from MFCC data."""
    fig, ax = plt.subplots(figsize=figsize)

    # Normalize data for better visualization
    normalized_data = normalize_mfcc_for_image(mfcc_data)

    # Create the image (transpose so time is x-axis, MFCC coefficients are y-axis)
    im = ax.imshow(
        normalized_data.T, aspect="auto", origin="lower", cmap="viridis", interpolation="nearest"
    )

    # Set labels and title
    ax.set_xlabel("Time Frames")
    ax.set_ylabel("MFCC Coefficients")
    ax.set_title(title)

    # Add colorbar
    plt.colorbar(im, ax=ax, label="Normalized MFCC Value")

    return fig


def save_sample_images(
    features: np.ndarray,
    labels: np.ndarray,
    genre_names: List[str],
    output_dir: str,
    samples_per_genre: int = 3,
) -> None:
    """Save sample images for each genre."""
    logger.info(f"Creating sample images for {len(genre_names)} genres")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for genre_idx, genre_name in enumerate(genre_names):
        # Find samples of this genre
        genre_samples = np.where(labels == genre_idx)[0]

        if len(genre_samples) == 0:
            logger.warning(f"No samples found for genre: {genre_name}")
            continue

        logger.info(f"Creating images for {genre_name} ({len(genre_samples)} samples available)")

        # Create sample images
        for i, sample_idx in enumerate(genre_samples[:samples_per_genre]):
            mfcc_data = features[sample_idx]

            # Create image
            fig = create_mfcc_image(
                mfcc_data, title=f"{genre_name} - Sample {i+1} (Index {sample_idx})"
            )

            # Save image
            image_path = output_path / f"{genre_name.lower()}_sample_{i+1}.png"
            fig.savefig(image_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            logger.info(f"Saved: {image_path}")


def create_genre_comparison_grid(
    features: np.ndarray,
    labels: np.ndarray,
    genre_names: List[str],
    output_dir: str,
    samples_per_genre: int = 2,
) -> None:
    """Create a grid comparing MFCC patterns across genres."""
    logger.info("Creating genre comparison grid")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Calculate grid dimensions
    n_genres = len(genre_names)
    n_cols = min(4, n_genres)  # Max 4 columns
    n_rows = (n_genres + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for genre_idx, genre_name in enumerate(genre_names):
        row = genre_idx // n_cols
        col = genre_idx % n_cols
        ax = axes[row, col]

        # Find samples of this genre
        genre_samples = np.where(labels == genre_idx)[0]

        if len(genre_samples) == 0:
            ax.text(
                0.5,
                0.5,
                f"No samples\nfor {genre_name}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(genre_name)
            continue

        # Use first sample for comparison
        sample_idx = genre_samples[0]
        mfcc_data = features[sample_idx]

        # Normalize for visualization
        normalized_data = normalize_mfcc_for_image(mfcc_data)

        # Create subplot
        im = ax.imshow(
            normalized_data.T,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            interpolation="nearest",
        )

        ax.set_title(f"{genre_name}")
        ax.set_xlabel("Time")
        ax.set_ylabel("MFCC")

        # Remove ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused subplots
    for genre_idx in range(n_genres, n_rows * n_cols):
        row = genre_idx // n_cols
        col = genre_idx % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()

    # Save comparison grid
    grid_path = output_path / "genre_comparison_grid.png"
    plt.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved genre comparison grid: {grid_path}")


def create_statistics_summary(
    features: np.ndarray, labels: np.ndarray, genre_names: List[str], output_dir: str
) -> None:
    """Create statistical summary of MFCC data."""
    logger.info("Creating statistics summary")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Calculate statistics
    stats = {
        "total_samples": len(features),
        "mfcc_shape": features.shape,
        "time_frames": features.shape[1],
        "mfcc_coefficients": features.shape[2],
        "genres": {},
    }

    for genre_idx, genre_name in enumerate(genre_names):
        genre_samples = np.where(labels == genre_idx)[0]

        if len(genre_samples) > 0:
            genre_features = features[genre_samples]
            stats["genres"][genre_name] = {
                "sample_count": len(genre_samples),
                "mean_mfcc": np.mean(genre_features, axis=(0, 1)).tolist(),
                "std_mfcc": np.std(genre_features, axis=(0, 1)).tolist(),
                "mean_energy": np.mean(genre_features[:, :, 0]).tolist(),  # First MFCC coefficient
            }

    # Save statistics
    stats_path = output_path / "mfcc_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Saved statistics: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert MFCC data to 2D images")
    parser.add_argument("--input", required=True, help="Path to MFCC JSON file")
    parser.add_argument("--output", required=True, help="Output directory for images")
    parser.add_argument(
        "--samples-per-genre",
        type=int,
        default=3,
        help="Number of sample images per genre (default: 3)",
    )
    parser.add_argument("--create-grid", action="store_true", help="Create genre comparison grid")
    parser.add_argument("--create-stats", action="store_true", help="Create statistics summary")

    args = parser.parse_args()

    # Load MFCC data
    features, labels, genre_names = load_mfcc_data(args.input)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing {len(features)} samples from {len(genre_names)} genres")
    logger.info(f"MFCC shape: {features.shape}")
    logger.info(f"Genres: {genre_names}")

    # Create sample images
    save_sample_images(features, labels, genre_names, str(output_dir), args.samples_per_genre)

    # Create genre comparison grid
    if args.create_grid:
        create_genre_comparison_grid(features, labels, genre_names, str(output_dir))

    # Create statistics summary
    if args.create_stats:
        create_statistics_summary(features, labels, genre_names, str(output_dir))

    logger.info("MFCC to image conversion completed!")


if __name__ == "__main__":
    main()
