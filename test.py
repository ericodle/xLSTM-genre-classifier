#!/usr/bin/env python3
"""
Master Test Runner

This script runs all tests in the tests/ directory.
Run with: python test.py
"""

import sys
import subprocess
from pathlib import Path


def run_linter(linter_name, args):
    """Run a linter and return its exit code."""
    print(f"\n{'=' * 70}")
    print(f"Running {linter_name}...")
    print(f"{'=' * 70}\n")
    
    try:
        result = subprocess.run(args, cwd=Path(__file__).parent.absolute())
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
    linter_results.append(run_linter(
        "isort (import sorting)",
        [sys.executable, "-m", "isort", "--check-only", "src/", "tests/"]
    ))
    
    # 2. black (code formatting)
    linter_results.append(run_linter(
        "black (code formatter)",
        [sys.executable, "-m", "black", "--check", "src/", "tests/"]
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
    
    pytest_args = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        # "-x",  # Uncomment to stop at first failure
    ]
    
    # Run pytest
    try:
        result = subprocess.run(pytest_args, cwd=project_root)
        
        print()
        print("=" * 70)
        if result.returncode == 0:
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed!")
        print("=" * 70)
        
        return result.returncode
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

