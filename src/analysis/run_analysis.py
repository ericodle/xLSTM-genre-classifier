#!/usr/bin/env python3
"""
Comprehensive Experiment Analysis Suite

This script orchestrates the complete analysis workflow for multiple training runs:
1. Aggregate results from all training runs
2. Analyze overfitting patterns in best models
3. Filter to show only best performing models

This provides a unified entry point for comprehensive experiment analysis.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# Handle both direct execution and module import
try:
    from .utils import AnalysisLogger, ensure_output_directory
except ImportError:
    # For direct execution, add the parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from analysis.utils import AnalysisLogger, ensure_output_directory

# Initialize logger
logger = AnalysisLogger("experiment_analysis")


def run_script(script_path: str, args: List[str], description: str) -> bool:
    """Run a Python script with given arguments."""
    logger.info(f"Running {description}...")

    try:
        cmd = [sys.executable, script_path] + args
        logger.debug(f"Command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        logger.info(f"✅ {description} completed successfully")
        if result.stdout:
            logger.debug(f"Output: {result.stdout}")

        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed with exit code {e.returncode}")
        if e.stdout:
            logger.error(f"STDOUT: {e.stdout}")
        if e.stderr:
            logger.error(f"STDERR: {e.stderr}")
        return False

    except Exception as e:
        logger.error(f"❌ {description} failed with error: {e}")
        return False


def run_complete_analysis(
    input_dir: str = "./outputs",
    output_dir: str = "./outputs/analysis",
    skip_aggregation: bool = False,
    skip_overfitting: bool = False,
    skip_filtering: bool = False,
    verbose: bool = False,
) -> bool:
    """
    Run the complete experiment analysis workflow.

    Args:
        input_dir: Directory containing training results
        output_dir: Directory for analysis outputs
        skip_aggregation: Skip results aggregation step
        skip_overfitting: Skip overfitting analysis step
        skip_filtering: Skip best models filtering step
        verbose: Enable verbose logging

    Returns:
        True if all steps completed successfully, False otherwise
    """

    # Set up logging level
    if verbose:
        logger.logger.setLevel(10)  # DEBUG level

    logger.info("=" * 80)
    logger.info("COMPREHENSIVE EXPERIMENT ANALYSIS SUITE")
    logger.info("=" * 80)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Ensure output directory exists
    ensure_output_directory(output_dir)

    # Get script directory
    script_dir = Path(__file__).parent
    success_count = 0
    total_steps = 3

    # Step 1: Aggregate Results
    if not skip_aggregation:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 1: AGGREGATING TRAINING RESULTS")
        logger.info("=" * 60)

        analyze_script = script_dir / "analyze_results.py"
        args = ["--input-dir", input_dir, "--output-dir", output_dir]

        if run_script(str(analyze_script), args, "Results Aggregation"):
            success_count += 1
        else:
            logger.error("Results aggregation failed. Continuing with remaining steps...")
    else:
        logger.info("Skipping results aggregation (--skip-aggregation)")
        success_count += 1

    # Step 2: Overfitting Analysis
    if not skip_overfitting:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: ANALYZING OVERFITTING PATTERNS")
        logger.info("=" * 60)

        overfitting_script = script_dir / "overfitting_analysis.py"
        args = ["--input-dir", str(input_dir), "--output-dir", str(output_dir)]

        if run_script(str(overfitting_script), args, "Overfitting Analysis"):
            success_count += 1
        else:
            logger.error("Overfitting analysis failed. Continuing with remaining steps...")
    else:
        logger.info("Skipping overfitting analysis (--skip-overfitting)")
        success_count += 1

    # Step 3: Filter Best Models
    if not skip_filtering:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: FILTERING TO BEST MODELS")
        logger.info("=" * 60)

        filter_script = script_dir / "filter_best_models.py"
        args = []  # filter_best_models.py doesn't take arguments

        if run_script(str(filter_script), args, "Best Models Filtering"):
            success_count += 1
        else:
            logger.error("Best models filtering failed.")
    else:
        logger.info("Skipping best models filtering (--skip-filtering)")
        success_count += 1

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Completed {success_count}/{total_steps} steps successfully")

    if success_count == total_steps:
        logger.info("🎉 All analysis steps completed successfully!")
        logger.info(f"📁 Results saved to: {output_dir}")
        logger.info("\nGenerated files:")

        # List generated files
        output_path = Path(output_dir)
        if output_path.exists():
            for file_path in sorted(output_path.glob("*.csv")):
                logger.info(f"  📊 {file_path.name}")
            for file_path in sorted(output_path.glob("*.png")):
                logger.info(f"  📈 {file_path.name}")

        return True
    else:
        logger.error(f"⚠️  {total_steps - success_count} step(s) failed")
        return False


def main() -> int:
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Experiment Analysis Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete analysis
  python src/analysis/run_analysis.py

  # Run with custom directories
  python src/analysis/run_analysis.py --input-dir ./my-outputs --output-dir ./my-analysis

  # Skip specific steps
  python src/analysis/run_analysis.py --skip-aggregation --skip-overfitting

  # Run with verbose logging
  python src/analysis/run_analysis.py --verbose

Workflow:
  1. Aggregate results from all training runs (analyze_results.py)
  2. Analyze overfitting patterns in best models (overfitting_analysis.py)  
  3. Filter to show only best performing models (filter_best_models.py)
        """,
    )

    parser.add_argument(
        "--input-dir",
        default="./outputs",
        help="Directory containing training results to analyze (default: ./outputs)",
    )

    parser.add_argument(
        "--output-dir",
        default="./outputs/analysis",
        help="Directory where analysis results will be saved (default: ./outputs/analysis)",
    )

    parser.add_argument(
        "--skip-aggregation", action="store_true", help="Skip the results aggregation step"
    )

    parser.add_argument(
        "--skip-overfitting", action="store_true", help="Skip the overfitting analysis step"
    )

    parser.add_argument(
        "--skip-filtering", action="store_true", help="Skip the best models filtering step"
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Validate input directory
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return 1

    # Run the complete analysis
    success = run_complete_analysis(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        skip_aggregation=args.skip_aggregation,
        skip_overfitting=args.skip_overfitting,
        skip_filtering=args.skip_filtering,
        verbose=args.verbose,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
