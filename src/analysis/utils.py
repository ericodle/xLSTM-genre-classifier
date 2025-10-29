"""
Shared utilities for analysis scripts.
Provides common functionality for data loading, plotting, and result saving.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Set up logging for analysis scripts."""
    logging.basicConfig(
        level=level, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)


def setup_plotting_style() -> None:
    """Set up consistent plotting style for analysis scripts."""
    # Set a clean theme with larger fonts for conference presentation
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.titlesize": 18,
            "axes.linewidth": 1.2,
            "grid.linewidth": 0.8,
            "lines.linewidth": 2,
            "patch.linewidth": 1.2,
        }
    )


def load_json_data(file_path: str) -> Dict[str, Any]:
    """Load JSON data from file with error handling."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {file_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading {file_path}: {e}")


def save_json_data(data: Dict[str, Any], file_path: str) -> None:
    """Save data to JSON file with error handling."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        raise RuntimeError(f"Error saving {file_path}: {e}")


def infer_dataset_from_path(path: str) -> str:
    """Infer dataset name from file path."""
    path_lower = path.lower()
    if "gtzan" in path_lower:
        return "GTZAN"
    elif "fma" in path_lower:
        return "FMA"
    else:
        return "UNKNOWN"


def infer_model_from_path(path: str, data: Optional[Dict[str, Any]] = None) -> str:
    """Infer model type from directory path or data."""
    # Try JSON fields first if data provided (most reliable)
    if data:
        # Check for model_type in various possible formats
        model_type = data.get("model_type")
        if model_type:
            # Normalize class names to standard type names
            if "ViT" in model_type or "Vision" in model_type:
                return "ViT"
            elif "VGG" in model_type or "vgg" in model_type.lower():
                return "VGG16"
            elif "Transformer" in model_type:
                return "TRANSFORMER"
            elif "xLSTM" in model_type:
                return "XLSTM"
            elif "GRU" in model_type:
                return "GRU"
            elif "LSTM" in model_type:
                return "LSTM"
            elif "CNN" in model_type:
                return "CNN"
            elif "FC" in model_type or "fc" in model_type.lower():
                return "FC"
            elif "SVM" in model_type:
                return "SVM"

        # SVM script stores params but not model string
        if "params" in data and "kernel" in data["params"]:
            return "SVM"

    # Try directory naming as fallback
    name = os.path.basename(path).lower()

    # Handle special cases
    if name.startswith("tr-") or "-tr-" in name or name.endswith("-tr"):
        return "TRANSFORMER"

    # Check for model types (order matters: check 'xlstm' before 'lstm', 'vit' before 'vgg')
    model_patterns = ["xlstm", "transformer", "vit", "vgg", "cnn", "lstm", "gru", "svm", "fc", "fv"]
    for pattern in model_patterns:
        if pattern in name:
            # Map VGG16 variations to consistent name
            if pattern == "vgg" or pattern == "fv":
                return "VGG16"
            elif pattern == "vit":
                return "ViT"
            else:
                return pattern.upper()

    return "UNKNOWN"


def ensure_output_directory(output_path: str) -> Path:
    """Ensure output directory exists."""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_plot(fig: plt.Figure, output_path: str, dpi: int = 300) -> None:
    """Save plot with consistent settings."""
    ensure_output_directory(os.path.dirname(output_path))
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


def save_dataframe(df: pd.DataFrame, output_path: str, index: bool = False) -> None:
    """Save DataFrame with consistent settings."""
    ensure_output_directory(os.path.dirname(output_path))
    df.to_csv(output_path, index=index)


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> None:
    """Validate DataFrame has required columns."""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame missing required columns: {missing_columns}")


def get_model_display_name(model: str) -> str:
    """Get display name for model type."""
    model_map = {
        "TRANSFORMER": "TR",
        "XLSTM": "XLSTM",
        "VGG16": "VGG",
        "VGG": "VGG",
    }
    return model_map.get(model, model)


def get_model_order() -> List[str]:
    """Get standard model order for plots."""
    return ["SVM", "FC", "CNN", "LSTM", "XLSTM", "GRU", "TR", "VGG"]


def get_dataset_colors() -> Dict[str, str]:
    """Get standard colors for datasets."""
    return {"GTZAN": "#2E86AB", "FMA": "#A23B72"}


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format value as percentage."""
    return f"{value * 100:.{decimals}f}%"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    return numerator / denominator if denominator != 0 else default


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """Calculate basic statistics for a list of values."""
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}

    values_array = np.array(values)
    return {
        "mean": float(np.mean(values_array)),
        "std": float(np.std(values_array)),
        "min": float(np.min(values_array)),
        "max": float(np.max(values_array)),
        "count": len(values),
    }


class AnalysisLogger:
    """Logger wrapper for analysis scripts."""

    def __init__(self, name: str = "analysis"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Add console handler if not already present
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)

    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)
