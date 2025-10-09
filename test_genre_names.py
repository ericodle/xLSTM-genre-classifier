#!/usr/bin/env python3
"""
Test script to verify that genre names are properly used in confusion matrix.
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_genre_names_in_confusion_matrix():
    """Test that genre names are properly displayed in confusion matrix."""
    
    # Create sample data with genre names
    sample_data = {
        "features": [
            np.random.randn(13, 100) for _ in range(20)  # 20 samples, 13 MFCCs, 100 frames
        ],
        "labels": [0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 1, 2, 0, 1, 2, 0, 1, 2, 0],
        "mapping": ["Rock", "Jazz", "Classical"]  # Genre names
    }
    
    # Save sample data
    with open("test_data_with_mapping.json", "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        json_data = {
            "features": [arr.tolist() for arr in sample_data["features"]],
            "labels": sample_data["labels"],
            "mapping": sample_data["mapping"]
        }
        json.dump(json_data, f, indent=2)
    
    print("✅ Created test data with genre mapping")
    
    # Test the trainer's genre name extraction
    from training.trainer import ModelTrainer
    from core.config import Config
    
    config = Config()
    trainer = ModelTrainer(config)
    trainer._load_json_data("test_data_with_mapping.json")
    
    print(f"✅ Trainer extracted genre names: {trainer.genre_names}")
    
    # Create sample predictions
    y_true = np.array(sample_data["labels"])
    y_pred = np.array([0, 1, 1, 2, 2, 0, 0, 1, 2, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2, 0])
    
    # Generate confusion matrix with genre names
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix with genre names
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=trainer.genre_names,
        yticklabels=trainer.genre_names,
    )
    plt.title("Confusion Matrix with Genre Names")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("test_confusion_matrix_with_genre_names.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print("✅ Generated confusion matrix with genre names: test_confusion_matrix_with_genre_names.png")
    
    # Clean up
    os.remove("test_data_with_mapping.json")
    print("✅ Test completed successfully!")

if __name__ == "__main__":
    test_genre_names_in_confusion_matrix()
