#!/usr/bin/env python3

import json
import numpy as np

# Load the MFCC data
with open('./mfccs/gtzan_13.json', 'r') as f:
    data = json.load(f)

print("Data keys:", list(data.keys()))
print("Features length:", len(data['features']))
print("Labels length:", len(data['labels']))

# Check first few features
for i in range(min(5, len(data['features']))):
    feature = data['features'][i]
    print(f"Feature {i}: shape = {len(feature)}")
    if feature:
        print(f"  First frame: {len(feature[0])} MFCCs")
        print(f"  Sample values: {feature[0][:3]}")

# Check if all features have the same number of frames
frame_counts = [len(feature) for feature in data['features']]
print(f"\nFrame count statistics:")
print(f"Min frames: {min(frame_counts)}")
print(f"Max frames: {max(frame_counts)}")
print(f"Mean frames: {np.mean(frame_counts):.2f}")
print(f"Std frames: {np.std(frame_counts):.2f}")

# Check unique labels
unique_labels = set(data['labels'])
print(f"\nUnique labels ({len(unique_labels)}): {sorted(unique_labels)}")

# Check label distribution
from collections import Counter
label_counts = Counter(data['labels'])
print(f"\nLabel distribution:")
for label, count in sorted(label_counts.items()):
    print(f"  {label}: {count}") 