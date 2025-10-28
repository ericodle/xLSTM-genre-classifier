#!/usr/bin/env python3
"""
Master Test Runner

This script runs all tests in the tests/ directory.
Run with: python test.py

Options:
  --fix      Auto-fix linter issues (runs isort and black without --check)
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_linter(linter_name, args):
    """Run a linter and return its exit code."""
    print(f"\n{'=' * 70}")
    print(f"Running {linter_name}...")
    print(f"{'=' * 70}\n")
    
    try:
        result = subprocess.run(
            args, 
            cwd=Path(__file__).parent.absolute(),
            capture_output=True,
            text=True
        )
        
        # Print the output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"✅ {linter_name} passed!\n")
        else:
            print(f"❌ {linter_name} found issues!\n")
        return result.returncode
    except Exception as e:
        print(f"❌ Error running {linter_name}: {e}\n")
        return 1


def main():
    """Run all tests and linters."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run tests and linters")
    parser.add_argument('--fix', action='store_true', help='Auto-fix linter issues')
    parser.add_argument('--training', action='store_true', help='Include heavy training tests')
    args = parser.parse_args()
    
    # Ensure we're in the project root
    project_root = Path(__file__).parent.absolute()
    
    print("=" * 70)
    print("Running All Checks (Tests + Linters)")
    print("=" * 70)
    print(f"Project root: {project_root}")
    print(f"Test directory: {project_root / 'tests'}")
    print("=" * 70)
    
    # Run linters first
    linter_results = []
    
    # 1. isort (import sorting)
    isort_args = [sys.executable, "-m", "isort"]
    if not args.fix:
        isort_args.append("--check-only")
    isort_args.extend(["src/", "tests/"])
    
    linter_results.append(run_linter(
        "isort (import sorting)" + (" - FIXING" if args.fix else ""),
        isort_args
    ))
    
    # 2. black (code formatting)
    black_args = [sys.executable, "-m", "black"]
    if not args.fix:
        black_args.append("--check")
    black_args.extend(["src/", "tests/"])
    
    linter_results.append(run_linter(
        "black (code formatter)" + (" - FIXING" if args.fix else ""),
        black_args
    ))
    
    # 3. mypy (type checking) - skip for now as it requires more setup
    # linter_results.append(run_linter(
    #     "mypy (type checker)",
    #     [sys.executable, "-m", "mypy", "src/"]
    # ))
    
    # Run pytest
    print(f"{'=' * 70}")
    print("Running pytest...")
    print(f"{'=' * 70}\n")
    
    base_args = [
        sys.executable, "-m", "pytest",
        "-vv",  # Extra verbose: show all tests and their statuses
        "--tb=short",  # Short traceback format
        "-ra",  # Show summary of skipped/xfailed/xpassed tests
    ]
    if args.training:
        pytest_args = base_args + ["tests/"]
    else:
        # Exclude heavy training tests directory by default
        pytest_args = base_args + ["tests", "-k", "not training"]
    
    # Run pytest
    try:
        result = subprocess.run(pytest_args, cwd=project_root, capture_output=False)
        
        # Check final status
        print()
        print("=" * 70)
        
        all_linters_passed = all(code == 0 for code in linter_results)
        tests_passed = (result.returncode == 0)
        
        if all_linters_passed and tests_passed:
            print("✅ All checks passed! (Linters + Tests)")
        elif tests_passed:
            print("⚠️  Tests passed but linters found issues")
            if not args.fix:
                print("\n💡 To auto-fix linter issues, run:")
                print("   python test.py --fix")
        elif all_linters_passed:
            print("⚠️  Linters passed but tests failed")
            print("\n💡 Check the test output above for details.")
        else:
            print("❌ Both linters and tests failed!")
            print("\n💡 Try fixing linters first with: python test.py --fix")
        
        print("=" * 70)
        
        # Return 0 only if everything passed
        return 0 if (all_linters_passed and tests_passed) else 1
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

