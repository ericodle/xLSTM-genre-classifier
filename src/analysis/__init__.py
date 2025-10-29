"""
Analysis module for GenreDiscern.

This module provides tools for analyzing training results, generating performance metrics,
and creating comparative visualizations across multiple training runs.
"""

from .analyze_results import collect_results, plot_bars, plot_model_grid
from .filter_best_models import filter_overfitting_analysis, get_best_models
from .overfitting_analysis import create_final_formatted_table, create_overfitting_analysis
from .run_analysis import run_complete_analysis
from .utils import (
    AnalysisLogger,
    calculate_statistics,
    ensure_output_directory,
    format_percentage,
    get_dataset_colors,
    get_model_display_name,
    get_model_order,
    infer_dataset_from_path,
    infer_model_from_path,
    load_json_data,
    safe_divide,
    save_json_data,
    setup_logging,
    setup_plotting_style,
)

__all__ = [
    # Main orchestrator
    "run_complete_analysis",
    # Individual analysis functions
    "collect_results",
    "plot_bars",
    "plot_model_grid",
    "filter_overfitting_analysis",
    "get_best_models",
    "create_overfitting_analysis",
    "create_final_formatted_table",
    # Utilities
    "AnalysisLogger",
    "calculate_statistics",
    "ensure_output_directory",
    "format_percentage",
    "get_dataset_colors",
    "get_model_display_name",
    "get_model_order",
    "infer_dataset_from_path",
    "infer_model_from_path",
    "load_json_data",
    "safe_divide",
    "save_json_data",
    "setup_logging",
    "setup_plotting_style",
]
