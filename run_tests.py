#!/usr/bin/env python3
"""
Simple test runner for GenreDiscern.
"""

import sys
import subprocess
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ SUCCESS")
        if result.stdout:
            print("Output:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print(f"Error code: {e.returncode}")
        if e.stdout:
            print("Stdout:")
            print(e.stdout)
        if e.stderr:
            print("Stderr:")
            print(e.stderr)
        return False


def main():
    """Main test runner function."""
    print("🎵 GenreDiscern Test Runner")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path("src").exists() or not Path("tests").exists():
        print("❌ Error: Please run this script from the GenreDiscern root directory")
        print("   Expected to find 'src/' and 'tests/' directories")
        sys.exit(1)
    
    # Check if pytest is available
    try:
        import pytest
        print(f"✅ pytest version: {pytest.__version__}")
    except ImportError:
        print("❌ Error: pytest not found. Please install it first:")
        print("   pip install -r requirements-test.txt")
        sys.exit(1)
    
    # Run tests
    success = True
    
    # Basic tests
    success &= run_command(
        ["python", "-m", "pytest", "tests/", "-v", "--tb=short"],
        "Basic test suite"
    )
    
    # Tests with coverage
    success &= run_command(
        ["python", "-m", "pytest", "tests/", "--cov=src", "--cov-report=term-missing"],
        "Tests with coverage report"
    )
    
    # Type checking (if mypy is available)
    try:
        import mypy
        success &= run_command(
            ["python", "-m", "mypy", "src/"],
            "Type checking with mypy"
        )
    except ImportError:
        print("⚠️  mypy not available, skipping type checking")
    
    # Code formatting check (if black is available)
    try:
        import black
        success &= run_command(
            ["python", "-m", "black", "--check", "src/", "tests/"],
            "Code formatting check with black"
        )
    except ImportError:
        print("⚠️  black not available, skipping formatting check")
    
    # Import tests
    success &= run_command(
        ["python", "-c", "import src; print('✅ All imports successful')"],
        "Import tests"
    )
    
    # Summary
    print(f"\n{'='*60}")
    if success:
        print("🎉 All tests passed successfully!")
        print("✅ Your GenreDiscern installation is working correctly")
    else:
        print("❌ Some tests failed. Please check the output above")
        print("💡 You may need to install dependencies or fix code issues")
    print('='*60)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 