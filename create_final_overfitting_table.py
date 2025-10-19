#!/usr/bin/env python3
"""
Create final overfitting analysis table in the specified order.

This script creates a single CSV file with rows in the exact order specified:
FC-FMA, FC-GTZAN, CNN-FMA, CNN-GTZAN, etc.
"""

import pandas as pd
import numpy as np

def get_best_models():
    """Get the best performing model for each model type and dataset."""
    
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

def create_final_table():
    """Create the final overfitting table in the specified order."""
    
    # Get best models mapping
    best_model_mapping, best_models_df = get_best_models()
    
    # Read the overfitting analysis
    overfitting_df = pd.read_csv('outputs/analysis/overfitting_analysis.csv')
    
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
    print("Creating final overfitting analysis table in specified order...")
    
    # Create the final table
    final_df = create_final_table()
    
    # Print the table
    print_final_table(final_df)
    
    # Save to CSV
    output_file = 'outputs/analysis/overfitting_analysis_final.csv'
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

if __name__ == "__main__":
    main()

