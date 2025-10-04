#!/usr/bin/env python3
"""
Run OFAT analysis for all model types (CNN, GRU, Transformer).
"""

import subprocess
import sys
import os
from datetime import datetime
import argparse


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"⏹️  {description} interrupted by user")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run OFAT analysis for all model types")
    parser.add_argument("--data", default="./mfccs/fma_13.json", 
                       help="Path to data file (default: ./mfccs/fma_13.json)")
    parser.add_argument("--output-base", default="./output", 
                       help="Base output directory (default: ./output)")
    parser.add_argument("--models", nargs="+", default=["CNN", "FC", "LSTM", "GRU", "Transformer", "xLSTM"],
                       choices=["CNN", "FC", "LSTM", "GRU", "Transformer", "xLSTM"],
                       help="Models to run OFAT for (default: CNN FC LSTM GRU Transformer xLSTM)")
    parser.add_argument("--skip-existing", action="store_true",
                       help="Skip models that already have output directories")
    
    args = parser.parse_args()
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("🎵 GenreDiscern - All Models OFAT Analysis")
    print("=" * 60)
    print(f"Data file: {args.data}")
    print(f"Output base: {args.output_base}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Timestamp: {timestamp}")
    print()
    
    # Check if data file exists
    if not os.path.exists(args.data):
        print(f"❌ Data file not found: {args.data}")
        sys.exit(1)
    
    # Track results
    results = {}
    total_models = len(args.models)
    
    for i, model in enumerate(args.models, 1):
        print(f"\n📊 Model {i}/{total_models}: {model}")
        
        # Create output directory
        output_dir = os.path.join(args.output_base, f"{model.lower()}_fma_{timestamp}")
        
        # Check if output already exists
        if args.skip_existing and os.path.exists(output_dir):
            print(f"⏭️  Skipping {model} - output directory already exists: {output_dir}")
            results[model] = "skipped"
            continue
        
        # Build command
        cmd = [
            "python", "run_ofat_analysis.py",
            "--model", model,
            "--data", args.data,
            "--output", output_dir
        ]
        
        # Run OFAT analysis
        success = run_command(cmd, f"OFAT Analysis for {model}")
        results[model] = "success" if success else "failed"
        
        if not success:
            print(f"⚠️  {model} failed - continuing with next model...")
    
    # Print summary
    print(f"\n{'='*60}")
    print("📋 SUMMARY")
    print(f"{'='*60}")
    
    for model, status in results.items():
        if status == "success":
            print(f"✅ {model}: Completed successfully")
        elif status == "skipped":
            print(f"⏭️  {model}: Skipped (already exists)")
        else:
            print(f"❌ {model}: Failed")
    
    # Count successes
    successful = sum(1 for status in results.values() if status == "success")
    total_run = sum(1 for status in results.values() if status != "skipped")
    
    print(f"\n📊 Results: {successful}/{total_run} models completed successfully")
    
    if successful == total_run and total_run > 0:
        print("🎉 All OFAT analyses completed successfully!")
        return 0
    else:
        print("⚠️  Some analyses failed or were skipped")
        return 1


if __name__ == "__main__":
    sys.exit(main())
