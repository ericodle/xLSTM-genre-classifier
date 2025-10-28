#!/usr/bin/env python3
"""
Master Test Runner

This script runs all tests in the tests/ directory.
Run with: python test.py
"""

import sys
import subprocess
from pathlib import Path


def main():
    """Run all tests in the tests directory."""
    # Ensure we're in the project root
    project_root = Path(__file__).parent.absolute()
    
    print("=" * 70)
    print("Running All Tests")
    print("=" * 70)
    print(f"Project root: {project_root}")
    print(f"Test directory: {project_root / 'tests'}")
    print("=" * 70)
    print()
    
    # Build pytest command
    # Use -v for verbose, --tb=short for cleaner tracebacks
    # Use -x to stop at first failure (optional, commented out by default)
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

