#!/usr/bin/env python3
"""
Format overfitting analysis table for scientific paper publication.

This script:
1. Reads the overfitting analysis CSV
2. Formats it with proper scientific notation and rounding
3. Creates a publication-ready table
4. Exports to both CSV and LaTeX formats
"""

import pandas as pd
import numpy as np
import os

def format_overfitting_table():
    """Format the overfitting analysis table for scientific publication."""
    
    # Read the original CSV
    df = pd.read_csv('outputs/analysis/overfitting_analysis.csv')
    
    # Round values to appropriate decimal places
    df['train_acc'] = df['train_acc'].round(3)
    df['val_acc'] = df['val_acc'].round(3)
    df['overfitting'] = df['overfitting'].round(3)
    
    # Format test accuracy (handle NaN values)
    df['test_acc_formatted'] = df['test_acc'].apply(
        lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
    )
    
    # Create a more readable format
    df_formatted = df.copy()
    
    # Add percentage signs and format for readability
    df_formatted['Train Acc (%)'] = (df['train_acc'] * 100).round(1).astype(str) + '%'
    df_formatted['Val Acc (%)'] = (df['val_acc'] * 100).round(1).astype(str) + '%'
    df_formatted['Overfitting (%)'] = (df['overfitting'] * 100).round(1).astype(str) + '%'
    
    # Create a clean table for publication
    publication_table = df_formatted[['model', 'dataset', 'epochs', 
                                    'Train Acc (%)', 'Val Acc (%)', 
                                    'Overfitting (%)']].copy()
    
    # Rename columns for publication
    publication_table.columns = ['Model', 'Dataset', 'Epochs', 
                               'Training Accuracy', 'Validation Accuracy', 
                               'Overfitting Gap']
    
    # Sort by overfitting gap (ascending - least overfitting first)
    publication_table = publication_table.sort_values('Overfitting Gap', 
                                                    key=lambda x: x.str.replace('%', '').astype(float))
    
    # Reset index
    publication_table = publication_table.reset_index(drop=True)
    
    # Add rank column
    publication_table.insert(0, 'Rank', range(1, len(publication_table) + 1))
    
    return publication_table, df

def create_latex_table(df_formatted):
    """Create a LaTeX table for scientific papers."""
    
    latex_content = """
\\begin{table}[htbp]
\\centering
\\caption{Overfitting Analysis: Training vs Validation Accuracy Gap}
\\label{tab:overfitting_analysis}
\\begin{tabular}{@{}lcccccc@{}}
\\toprule
\\textbf{Rank} & \\textbf{Model} & \\textbf{Dataset} & \\textbf{Epochs} & \\textbf{Train Acc} & \\textbf{Val Acc} & \\textbf{Overfitting Gap} \\\\
\\midrule
"""
    
    for _, row in df_formatted.iterrows():
        train_acc = row['Training Accuracy'].replace('%', '\\%')
        val_acc = row['Validation Accuracy'].replace('%', '\\%')
        overfitting = row['Overfitting Gap'].replace('%', '\\%')
        
        latex_content += f"{row['Rank']} & {row['Model']} & {row['Dataset']} & {row['Epochs']} & {train_acc} & {val_acc} & {overfitting} \\\\\n"
    
    latex_content += """
\\bottomrule
\\end{tabular}
\\begin{tablenotes}
\\small
\\item Note: Overfitting Gap = Training Accuracy - Validation Accuracy. 
Models are ranked by overfitting gap (ascending order).
\\end{tablenotes}
\\end{table}
"""
    
    return latex_content

def create_summary_statistics(df):
    """Create summary statistics for the paper."""
    
    # Calculate statistics
    stats = {
        'Total Models': len(df),
        'Average Overfitting': f"{df['overfitting'].mean() * 100:.1f}%",
        'Median Overfitting': f"{df['overfitting'].median() * 100:.1f}%",
        'Max Overfitting': f"{df['overfitting'].max() * 100:.1f}%",
        'Min Overfitting': f"{df['overfitting'].min() * 100:.1f}%",
        'Std Dev Overfitting': f"{df['overfitting'].std() * 100:.1f}%"
    }
    
    # Find best and worst models
    best_model = df.loc[df['overfitting'].idxmin()]
    worst_model = df.loc[df['overfitting'].idxmax()]
    
    stats['Best Model (Least Overfitting)'] = f"{best_model['model']} ({best_model['dataset']}) - {best_model['overfitting'] * 100:.1f}%"
    stats['Worst Model (Most Overfitting)'] = f"{worst_model['model']} ({worst_model['dataset']}) - {worst_model['overfitting'] * 100:.1f}%"
    
    return stats

def main():
    """Main function to format the overfitting table."""
    
    print("Formatting overfitting analysis table for scientific publication...")
    
    # Create formatted table
    publication_table, original_df = format_overfitting_table()
    
    # Save formatted CSV
    output_file = 'outputs/analysis/overfitting_analysis_formatted.csv'
    publication_table.to_csv(output_file, index=False)
    print(f"Formatted table saved to: {output_file}")
    
    # Create LaTeX table
    latex_content = create_latex_table(publication_table)
    latex_file = 'outputs/analysis/overfitting_analysis.tex'
    with open(latex_file, 'w') as f:
        f.write(latex_content)
    print(f"LaTeX table saved to: {latex_file}")
    
    # Create summary statistics
    stats = create_summary_statistics(original_df)
    
    # Save summary statistics
    stats_file = 'outputs/analysis/overfitting_summary.txt'
    with open(stats_file, 'w') as f:
        f.write("OVERFITTING ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    print(f"Summary statistics saved to: {stats_file}")
    
    # Display the formatted table
    print("\n" + "="*100)
    print("FORMATTED OVERFITTING ANALYSIS TABLE")
    print("="*100)
    print(publication_table.to_string(index=False))
    
    print(f"\nSummary Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\nFiles created:")
    print(f"  - {output_file} (CSV format)")
    print(f"  - {latex_file} (LaTeX format)")
    print(f"  - {stats_file} (Summary statistics)")

if __name__ == "__main__":
    main()
