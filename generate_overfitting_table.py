#!/usr/bin/env python3
"""
Complete workflow to generate the final overfitting analysis table.

This script:
1. Generates the overfitting analysis from training metadata
2. Filters to best performing models only
3. Creates the final table in the specified order
"""

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path

def extract_model_info(run_dir):
    """Extract model information from directory name."""
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

def get_best_models():
    """Identify the best performing model for each model type and dataset."""
    
    # Read the results summary to get test accuracies
    results_df = pd.read_csv('outputs/analysis/results_summary.csv')
    
    # Remove rows with missing test_acc
    results_df = results_df.dropna(subset=['test_acc'])
    
    # Group by model and dataset, then find the one with highest test accuracy
    best_models = results_df.loc[results_df.groupby(['model', 'dataset'])['test_acc'].idxmax()]
    
    # Create a mapping of (model, dataset) -> run_dir
    best_model_mapping = {}
    for _, row in best_models.iterrows():
        key = (row['model'], row['dataset'])
        best_model_mapping[key] = row['run_dir']
    
    return best_model_mapping, best_models

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

def create_final_table():
    """Create the final overfitting table in the specified order."""
    
    print("Step 1: Analyzing overfitting from training metadata...")
    overfitting_results = analyze_overfitting()
    
    if not overfitting_results:
        print("No overfitting results found!")
        return None
    
    print("Step 2: Identifying best performing models...")
    best_model_mapping, best_models_df = get_best_models()
    
    # Convert to DataFrame
    overfitting_df = pd.DataFrame(overfitting_results)
    
    # Create a key for matching
    overfitting_df['key'] = list(zip(overfitting_df['model'], overfitting_df['dataset']))
    
    # Filter to only include best models
    best_overfitting = overfitting_df[overfitting_df['key'].isin(best_model_mapping.keys())].copy()
    
    # Remove the key column and duplicates
    best_overfitting = best_overfitting.drop('key', axis=1)
    best_overfitting = best_overfitting.drop_duplicates(subset=['model', 'dataset'])
    
    # Format percentages
    best_overfitting['Training Acc (%)'] = (best_overfitting['train_acc'] * 100).round(1)
    best_overfitting['Validation Acc (%)'] = (best_overfitting['val_acc'] * 100).round(1)
    best_overfitting['Overfitting Gap (%)'] = (best_overfitting['overfitting'] * 100).round(1)
    
    print("Step 3: Creating final table in specified order...")
    
    # Define the exact order you specified
    model_dataset_order = [
        ('FC', 'FMA'),
        ('FC', 'GTZAN'),
        ('CNN', 'FMA'),
        ('CNN', 'GTZAN'),
        ('LSTM', 'FMA'),
        ('LSTM', 'GTZAN'),
        ('XLSTM', 'FMA'),
        ('XLSTM', 'GTZAN'),
        ('GRU', 'FMA'),
        ('GRU', 'GTZAN'),
        ('TRANSFORMER', 'FMA'),
        ('TRANSFORMER', 'GTZAN'),
        ('VGG', 'FMA'),
        ('VGG', 'GTZAN')
    ]
    
    # Create empty list to store rows
    final_rows = []
    
    # Process each model-dataset combination in the specified order
    for model, dataset in model_dataset_order:
        # Find the row for this model-dataset combination
        model_row = best_overfitting[(best_overfitting['model'] == model) & (best_overfitting['dataset'] == dataset)]
        
        if not model_row.empty:
            # Get the data
            row_data = {
                'Model': model,
                'Dataset': dataset,
                'Training Acc (%)': model_row['Training Acc (%)'].iloc[0],
                'Validation Acc (%)': model_row['Validation Acc (%)'].iloc[0],
                'Overfitting Gap (%)': model_row['Overfitting Gap (%)'].iloc[0]
            }
        else:
            # Create empty row for missing data
            row_data = {
                'Model': model,
                'Dataset': dataset,
                'Training Acc (%)': 'N/A',
                'Validation Acc (%)': 'N/A',
                'Overfitting Gap (%)': 'N/A'
            }
        
        final_rows.append(row_data)
    
    # Create final dataframe
    final_df = pd.DataFrame(final_rows)
    
    return final_df

def print_final_table(df):
    """Print the final table."""
    print(f"\n{'='*100}")
    print("FINAL OVERFITTING ANALYSIS TABLE")
    print("(In specified order)")
    print(f"{'='*100}")
    
    # Print header
    header = f"{'Model':<12} {'Dataset':<6} {'Training Acc %':<15} {'Validation Acc %':<15} {'Overfitting %':<12}"
    print(header)
    print("-" * 100)
    
    # Print data rows
    for _, row in df.iterrows():
        train_acc = f"{row['Training Acc (%)']}" if row['Training Acc (%)'] != 'N/A' else "N/A"
        val_acc = f"{row['Validation Acc (%)']}" if row['Validation Acc (%)'] != 'N/A' else "N/A"
        overfitting = f"{row['Overfitting Gap (%)']}" if row['Overfitting Gap (%)'] != 'N/A' else "N/A"
        
        line = f"{row['Model']:<12} {row['Dataset']:<6} {train_acc:<15} {val_acc:<15} {overfitting:<12}"
        print(line)

def main():
    """Main function."""
    print("Generating overfitting analysis table...")
    
    # Create the final table
    final_df = create_final_table()
    
    if final_df is not None:
        # Print the table
        print_final_table(final_df)
        
        # Save to CSV
        output_file = 'outputs/analysis/overfitting_analysis_final.csv'
        os.makedirs('outputs/analysis', exist_ok=True)
        final_df.to_csv(output_file, index=False)
        print(f"\nFinal table saved to: {output_file}")
        
        # Print summary
        print(f"\n{'='*100}")
        print("SUMMARY")
        print(f"{'='*100}")
        print(f"Total rows: {len(final_df)}")
        print(f"Models included: {final_df['Model'].nunique()}")
        print(f"Datasets included: {final_df['Dataset'].nunique()}")
        
        # Count available data
        available_data = final_df[final_df['Overfitting Gap (%)'] != 'N/A']
        print(f"Rows with data: {len(available_data)}")
        print(f"Rows missing data: {len(final_df) - len(available_data)}")
    else:
        print("Failed to generate overfitting analysis table!")

if __name__ == "__main__":
    main()
