#!/usr/bin/env python3
"""
Generate overfitting analysis table using test accuracy from evaluation results.

This script:
1. Finds the best performing model for each model type and dataset based on test accuracy from evaluation results
2. Gets the training and validation accuracy from the last epoch of those best models
3. Calculates overfitting gap (train_acc - val_acc) for those best models only
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

def load_test_accuracy_from_evaluation(run_dir):
    """Load test accuracy from evaluation results."""
    eval_dir = Path(f"outputs/{run_dir}/evaluation")
    
    if not eval_dir.exists():
        return None
    
    # Look for evaluation metrics file
    metrics_file = eval_dir / "evaluation_metrics.txt"
    if metrics_file.exists():
        try:
            with open(metrics_file, 'r') as f:
                content = f.read()
                # Look for "Accuracy: X.XXXX" pattern
                for line in content.split('\n'):
                    if line.startswith('Accuracy:'):
                        accuracy_str = line.split(':')[1].strip()
                        return float(accuracy_str)
        except Exception as e:
            print(f"Error reading {metrics_file}: {e}")
    
    # Look for JSON evaluation results
    json_file = eval_dir / "evaluation_results.json"
    if json_file.exists():
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                if 'accuracy' in data:
                    return float(data['accuracy'])
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
    
    return None

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

def find_best_models():
    """Find the best performing model for each model type and dataset based on test accuracy."""
    outputs_dir = Path("outputs")
    model_results = {}
    
    # Find all model directories
    for run_dir in outputs_dir.iterdir():
        if run_dir.is_dir() and run_dir.name != "analysis":
            model, dataset = extract_model_info(run_dir.name)
            if not model or not dataset:
                continue
            
            # Get test accuracy from evaluation results
            test_acc = load_test_accuracy_from_evaluation(run_dir.name)
            
            if test_acc is not None:
                key = (model, dataset)
                if key not in model_results or test_acc > model_results[key]['test_acc']:
                    model_results[key] = {
                        'run_dir': run_dir.name,
                        'model': model,
                        'dataset': dataset,
                        'test_acc': test_acc
                    }
                    print(f"Found {model} on {dataset}: {test_acc:.4f} test accuracy ({run_dir.name})")
    
    return model_results

def get_training_metrics_for_best_models(best_models):
    """Get training and validation accuracy for the best models."""
    results = []
    
    for key, model_info in best_models.items():
        run_dir = model_info['run_dir']
        model = model_info['model']
        dataset = model_info['dataset']
        test_acc = model_info['test_acc']
        
        # Look for metadata files
        metadata_file = Path(f"outputs/{run_dir}/model_metadata.json")
        if not metadata_file.exists():
            metadata_file = Path(f"outputs/{run_dir}/best_model_metadata.json")
        
        if metadata_file.exists():
            print(f"Getting training metrics for {model} on {dataset}...")
            
            # Load training history
            train_acc, val_acc, epochs = load_training_history(metadata_file)
            
            if train_acc is not None and val_acc is not None:
                overfitting = train_acc - val_acc
                results.append({
                    'run_dir': run_dir,
                    'model': model,
                    'dataset': dataset,
                    'test_acc': test_acc,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'overfitting': overfitting,
                    'epochs': epochs
                })
                print(f"  Train: {train_acc:.4f}, Val: {val_acc:.4f}, Overfitting: {overfitting:.4f}")
            else:
                print(f"  No training history found for {run_dir}")
        else:
            print(f"  No metadata file found for {run_dir}")
    
    return results

def create_final_table():
    """Create the final overfitting table in the specified order."""
    
    print("Step 1: Finding best performing models based on test accuracy...")
    best_models = find_best_models()
    
    if not best_models:
        print("No best models found!")
        return None
    
    print(f"\nFound {len(best_models)} best performing models:")
    for key, model_info in best_models.items():
        print(f"  {model_info['model']} on {model_info['dataset']}: {model_info['test_acc']:.4f}")
    
    print("\nStep 2: Getting training metrics for best models...")
    training_results = get_training_metrics_for_best_models(best_models)
    
    if not training_results:
        print("No training results found!")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(training_results)
    
    # Format percentages
    df['Training Acc (%)'] = (df['train_acc'] * 100).round(1)
    df['Validation Acc (%)'] = (df['val_acc'] * 100).round(1)
    df['Overfitting Gap (%)'] = (df['overfitting'] * 100).round(1)
    df['Test Acc (%)'] = (df['test_acc'] * 100).round(1)
    
    print("\nStep 3: Creating final table in specified order...")
    
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
        model_row = df[(df['model'] == model) & (df['dataset'] == dataset)]
        
        if not model_row.empty:
            # Get the data
            row_data = {
                'Model': model,
                'Dataset': dataset,
                'Test Acc (%)': model_row['Test Acc (%)'].iloc[0],
                'Training Acc (%)': model_row['Training Acc (%)'].iloc[0],
                'Validation Acc (%)': model_row['Validation Acc (%)'].iloc[0],
                'Overfitting Gap (%)': model_row['Overfitting Gap (%)'].iloc[0]
            }
        else:
            # Create empty row for missing data
            row_data = {
                'Model': model,
                'Dataset': dataset,
                'Test Acc (%)': 'N/A',
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
    print(f"\n{'='*120}")
    print("FINAL OVERFITTING ANALYSIS TABLE - BEST PERFORMING MODELS ONLY")
    print("(Based on test accuracy from evaluation results)")
    print(f"{'='*120}")
    
    # Print header
    header = f"{'Model':<12} {'Dataset':<6} {'Test Acc %':<12} {'Training Acc %':<15} {'Validation Acc %':<15} {'Overfitting %':<12}"
    print(header)
    print("-" * 120)
    
    # Print data rows
    for _, row in df.iterrows():
        test_acc = f"{row['Test Acc (%)']}" if row['Test Acc (%)'] != 'N/A' else "N/A"
        train_acc = f"{row['Training Acc (%)']}" if row['Training Acc (%)'] != 'N/A' else "N/A"
        val_acc = f"{row['Validation Acc (%)']}" if row['Validation Acc (%)'] != 'N/A' else "N/A"
        overfitting = f"{row['Overfitting Gap (%)']}" if row['Overfitting Gap (%)'] != 'N/A' else "N/A"
        
        line = f"{row['Model']:<12} {row['Dataset']:<6} {test_acc:<12} {train_acc:<15} {val_acc:<15} {overfitting:<12}"
        print(line)

def main():
    """Main function."""
    print("Generating overfitting analysis table using test accuracy from evaluation results...")
    
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
        print(f"\n{'='*120}")
        print("SUMMARY")
        print(f"{'='*120}")
        print(f"Total rows: {len(final_df)}")
        print(f"Models included: {final_df['Model'].nunique()}")
        print(f"Datasets included: {final_df['Dataset'].nunique()}")
        
        # Count available data
        available_data = final_df[final_df['Overfitting Gap (%)'] != 'N/A']
        print(f"Rows with data: {len(available_data)}")
        print(f"Rows missing data: {len(final_df) - len(available_data)}")
        
        # Show test accuracy range
        test_accs = final_df[final_df['Test Acc (%)'] != 'N/A']['Test Acc (%)']
        if len(test_accs) > 0:
            print(f"Test accuracy range: {test_accs.min():.1f}% to {test_accs.max():.1f}%")
    else:
        print("Failed to generate overfitting analysis table!")

if __name__ == "__main__":
    main()
