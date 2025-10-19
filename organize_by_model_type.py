#!/usr/bin/env python3
"""
Organize overfitting analysis tables by model type.

This script:
1. Uses the same model order as in test_acc_by_model_dataset.png
2. Creates tables with model type as the first column
3. Shows both FMA and GTZAN results for each model type
"""

import pandas as pd
import numpy as np

def organize_by_model_type():
    """Organize overfitting analysis by model type using the same order as visualization."""
    
    # Read the best models overfitting analysis
    fma_df = pd.read_csv('outputs/analysis/overfitting_analysis_FMA_best.csv')
    gtzan_df = pd.read_csv('outputs/analysis/overfitting_analysis_GTZAN_best.csv')
    
    # Define model order (same as in test_acc_by_model_dataset.png)
    MODEL_ORDER = ["SVM", "FC", "CNN", "LSTM", "XLSTM", "GRU", "TR", "VGG"]
    
    # Create a combined dataframe with dataset information
    fma_df['Dataset'] = 'FMA'
    gtzan_df['Dataset'] = 'GTZAN'
    
    # Combine dataframes
    combined_df = pd.concat([fma_df, gtzan_df], ignore_index=True)
    
    # Create a mapping for model order
    model_order_map = {model: i for i, model in enumerate(MODEL_ORDER)}
    
    # Add order column and sort
    combined_df['Model_Order'] = combined_df['Model'].map(model_order_map)
    combined_df = combined_df.sort_values('Model_Order').reset_index(drop=True)
    
    # Remove the order column and rank column
    combined_df = combined_df.drop(['Model_Order', 'Rank'], axis=1)
    
    # Reorder columns to put Model first
    combined_df = combined_df[['Model', 'Dataset', 'Training Acc (%)', 'Validation Acc (%)', 'Overfitting Gap (%)']]
    
    return combined_df, fma_df, gtzan_df

def print_organized_table(df):
    """Print the organized table by model type."""
    print(f"\n{'='*100}")
    print("OVERFITTING ANALYSIS - ORGANIZED BY MODEL TYPE")
    print("(Same order as test_acc_by_model_dataset.png)")
    print(f"{'='*100}")
    
    # Print header
    header = f"{'Model':<12} {'Dataset':<6} {'Training Acc %':<15} {'Validation Acc %':<15} {'Overfitting %':<12}"
    print(header)
    print("-" * 100)
    
    # Print data rows
    for _, row in df.iterrows():
        line = f"{row['Model']:<12} {row['Dataset']:<6} {row['Training Acc (%)']:<15} {row['Validation Acc (%)']:<15} {row['Overfitting Gap (%)']:<12}"
        print(line)
    
    # Print summary by model type
    print(f"\n{'='*100}")
    print("SUMMARY BY MODEL TYPE")
    print(f"{'='*100}")
    
    for model in ["SVM", "FC", "CNN", "LSTM", "XLSTM", "GRU", "TR", "VGG"]:
        model_data = df[df['Model'] == model]
        if not model_data.empty:
            print(f"\n{model}:")
            for _, row in model_data.iterrows():
                print(f"  {row['Dataset']}: {row['Training Acc (%)']}% train, {row['Validation Acc (%)']}% val, {row['Overfitting Gap (%)']}% overfitting")
        else:
            print(f"\n{model}: No data available")

def create_side_by_side_table(fma_df, gtzan_df):
    """Create a side-by-side comparison table."""
    print(f"\n{'='*120}")
    print("SIDE-BY-SIDE COMPARISON: FMA vs GTZAN")
    print(f"{'='*120}")
    
    # Create a mapping for model order
    MODEL_ORDER = ["SVM", "FC", "CNN", "LSTM", "XLSTM", "GRU", "TR", "VGG"]
    model_order_map = {model: i for i, model in enumerate(MODEL_ORDER)}
    
    # Get all unique models from both datasets
    all_models = set(fma_df['Model'].unique()) | set(gtzan_df['Model'].unique())
    all_models = sorted(all_models, key=lambda x: model_order_map.get(x, 999))
    
    # Print header
    header = f"{'Model':<12} {'FMA Overfitting %':<18} {'GTZAN Overfitting %':<20} {'Difference %':<15}"
    print(header)
    print("-" * 120)
    
    # Print data rows
    for model in all_models:
        fma_row = fma_df[fma_df['Model'] == model]
        gtzan_row = gtzan_df[gtzan_df['Model'] == model]
        
        fma_overfitting = fma_row['Overfitting Gap (%)'].iloc[0] if not fma_row.empty else "N/A"
        gtzan_overfitting = gtzan_row['Overfitting Gap (%)'].iloc[0] if not gtzan_row.empty else "N/A"
        
        if fma_overfitting != "N/A" and gtzan_overfitting != "N/A":
            difference = gtzan_overfitting - fma_overfitting
            diff_str = f"{difference:+.1f}"
        else:
            diff_str = "N/A"
        
        fma_str = f"{fma_overfitting}" if fma_overfitting != "N/A" else "N/A"
        gtzan_str = f"{gtzan_overfitting}" if gtzan_overfitting != "N/A" else "N/A"
        
        line = f"{model:<12} {fma_str:<18} {gtzan_str:<20} {diff_str:<15}"
        print(line)

def main():
    """Main function."""
    print("Organizing overfitting analysis by model type...")
    
    # Organize data by model type
    combined_df, fma_df, gtzan_df = organize_by_model_type()
    
    # Print organized table
    print_organized_table(combined_df)
    
    # Create side-by-side comparison
    create_side_by_side_table(fma_df, gtzan_df)
    
    # Save organized table
    combined_df.to_csv('outputs/analysis/overfitting_analysis_by_model_type.csv', index=False)
    print(f"\nOrganized table saved to: outputs/analysis/overfitting_analysis_by_model_type.csv")
    
    # Print summary statistics
    print(f"\n{'='*100}")
    print("SUMMARY STATISTICS")
    print(f"{'='*100}")
    print(f"FMA Dataset:")
    print(f"  Models: {len(fma_df)}")
    print(f"  Avg Overfitting: {fma_df['Overfitting Gap (%)'].mean():.1f}%")
    print(f"  Range: {fma_df['Overfitting Gap (%)'].min():.1f}% to {fma_df['Overfitting Gap (%)'].max():.1f}%")
    print(f"\nGTZAN Dataset:")
    print(f"  Models: {len(gtzan_df)}")
    print(f"  Avg Overfitting: {gtzan_df['Overfitting Gap (%)'].mean():.1f}%")
    print(f"  Range: {gtzan_df['Overfitting Gap (%)'].min():.1f}% to {gtzan_df['Overfitting Gap (%)'].max():.1f}%")

if __name__ == "__main__":
    main()
