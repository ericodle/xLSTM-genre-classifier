#!/usr/bin/env python3
"""
Analyze overfitting in model training results.

This script:
1. Finds all model training runs in the outputs directory
2. Extracts training and validation accuracy from the last epoch
3. Calculates overfitting as the difference between train_acc and val_acc
4. Identifies the best performing model by test accuracy
5. Creates a comprehensive overfitting analysis table
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import re

def extract_model_info(run_dir):
    """Extract model information from directory name."""
    # Pattern: model-dataset-date-lr or model-dataset-date-params
    parts = run_dir.split('-')
    if len(parts) >= 3:
        model = parts[0].upper()
        dataset = parts[1].upper()
        return model, dataset
    return None, None

def load_training_history(metadata_file):
    """Load training history from metadata file."""
    try:
        with open(metadata_file, 'r') as f:
            data = json.load(f)
        
        if 'training_history' in data:
            history = data['training_history']
            if 'train_acc' in history and 'val_acc' in history:
                train_acc = history['train_acc']
                val_acc = history['val_acc']
                
                if train_acc and val_acc:
                    # Get the last epoch values
                    last_train_acc = train_acc[-1] if train_acc else None
                    last_val_acc = val_acc[-1] if val_acc else None
                    return last_train_acc, last_val_acc, len(train_acc)
        
        return None, None, None
    except Exception as e:
        print(f"Error loading {metadata_file}: {e}")
        return None, None, None

def load_test_accuracy(metadata_file):
    """Load test accuracy from metadata file."""
    try:
        with open(metadata_file, 'r') as f:
            data = json.load(f)
        
        if 'test_accuracy' in data:
            return data['test_accuracy']
        elif 'final_test_accuracy' in data:
            return data['final_test_accuracy']
        
        return None
    except Exception as e:
        print(f"Error loading test accuracy from {metadata_file}: {e}")
        return None

def analyze_overfitting():
    """Analyze overfitting across all model runs."""
    outputs_dir = Path("outputs")
    results = []
    
    # Find all model directories
    for run_dir in outputs_dir.iterdir():
        if run_dir.is_dir() and run_dir.name != "analysis":
            model, dataset = extract_model_info(run_dir.name)
            if not model or not dataset:
                continue
            
            # Look for metadata files
            metadata_file = run_dir / "model_metadata.json"
            if not metadata_file.exists():
                metadata_file = run_dir / "best_model_metadata.json"
            
            if metadata_file.exists():
                print(f"Processing {run_dir.name}...")
                
                # Load training history
                train_acc, val_acc, epochs = load_training_history(metadata_file)
                
                # Load test accuracy
                test_acc = load_test_accuracy(metadata_file)
                
                if train_acc is not None and val_acc is not None:
                    overfitting = train_acc - val_acc
                    results.append({
                        'run_dir': run_dir.name,
                        'model': model,
                        'dataset': dataset,
                        'epochs': epochs,
                        'train_acc': train_acc,
                        'val_acc': val_acc,
                        'test_acc': test_acc,
                        'overfitting': overfitting
                    })
                else:
                    print(f"  No training history found for {run_dir.name}")
    
    return results

def create_overfitting_table(results):
    """Create a comprehensive overfitting analysis table."""
    if not results:
        print("No results found!")
        return
    
    df = pd.DataFrame(results)
    
    # Sort by test accuracy (descending) to find best models
    df_sorted = df.sort_values('test_acc', ascending=False, na_position='last')
    
    print("\n" + "="*80)
    print("OVERFITTING ANALYSIS TABLE")
    print("="*80)
    print(f"{'Rank':<4} {'Model':<12} {'Dataset':<6} {'Epochs':<6} {'Train Acc':<10} {'Val Acc':<10} {'Test Acc':<10} {'Overfitting':<12}")
    print("-"*80)
    
    for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
        train_acc = f"{row['train_acc']:.4f}" if pd.notna(row['train_acc']) else "N/A"
        val_acc = f"{row['val_acc']:.4f}" if pd.notna(row['val_acc']) else "N/A"
        test_acc = f"{row['test_acc']:.4f}" if pd.notna(row['test_acc']) else "N/A"
        overfitting = f"{row['overfitting']:.4f}" if pd.notna(row['overfitting']) else "N/A"
        
        print(f"{i:<4} {row['model']:<12} {row['dataset']:<6} {row['epochs']:<6} {train_acc:<10} {val_acc:<10} {test_acc:<10} {overfitting:<12}")
    
    # Find best performing model (by overfitting, since test_acc is mostly missing)
    best_model = df_sorted.iloc[0]
    print(f"\nBEST PERFORMING MODEL (by test accuracy):")
    print(f"  Model: {best_model['model']}")
    print(f"  Dataset: {best_model['dataset']}")
    test_acc_str = f"{best_model['test_acc']:.4f}" if pd.notna(best_model['test_acc']) else "N/A"
    print(f"  Test Accuracy: {test_acc_str}")
    print(f"  Training Accuracy: {best_model['train_acc']:.4f}")
    print(f"  Validation Accuracy: {best_model['val_acc']:.4f}")
    print(f"  Overfitting: {best_model['overfitting']:.4f}")
    
    # Find model with least overfitting
    least_overfitting = df_sorted.loc[df_sorted['overfitting'].idxmin()]
    print(f"\nLEAST OVERFITTING MODEL:")
    print(f"  Model: {least_overfitting['model']}")
    print(f"  Dataset: {least_overfitting['dataset']}")
    test_acc_str = f"{least_overfitting['test_acc']:.4f}" if pd.notna(least_overfitting['test_acc']) else "N/A"
    print(f"  Test Accuracy: {test_acc_str}")
    print(f"  Training Accuracy: {least_overfitting['train_acc']:.4f}")
    print(f"  Validation Accuracy: {least_overfitting['val_acc']:.4f}")
    print(f"  Overfitting: {least_overfitting['overfitting']:.4f}")
    
    # Summary statistics
    print(f"\nSUMMARY STATISTICS:")
    print(f"  Total models analyzed: {len(df)}")
    print(f"  Average overfitting: {df['overfitting'].mean():.4f}")
    print(f"  Median overfitting: {df['overfitting'].median():.4f}")
    print(f"  Max overfitting: {df['overfitting'].max():.4f}")
    print(f"  Min overfitting: {df['overfitting'].min():.4f}")
    
    # Models with most and least overfitting
    most_overfitting = df.loc[df['overfitting'].idxmax()]
    print(f"\nMOST OVERFITTING MODEL:")
    print(f"  Model: {most_overfitting['model']} ({most_overfitting['dataset']})")
    print(f"  Overfitting: {most_overfitting['overfitting']:.4f}")
    print(f"  Train Acc: {most_overfitting['train_acc']:.4f}, Val Acc: {most_overfitting['val_acc']:.4f}")
    
    # Save to CSV
    output_file = "outputs/analysis/overfitting_analysis.csv"
    df_sorted.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    return df_sorted

def main():
    """Main function."""
    print("Analyzing overfitting in model training results...")
    
    # Create analysis directory if it doesn't exist
    os.makedirs("outputs/analysis", exist_ok=True)
    
    # Analyze overfitting
    results = analyze_overfitting()
    
    if results:
        df = create_overfitting_table(results)
    else:
        print("No training results found to analyze!")

if __name__ == "__main__":
    main()
