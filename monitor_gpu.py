#!/usr/bin/env python3
"""
Simple GPU monitoring script for debugging training issues.
"""

import time
import subprocess
import sys

def get_gpu_info():
    """Get GPU usage information."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return "GPU info unavailable"
    except Exception as e:
        return f"Error: {e}"

def main():
    """Monitor GPU usage."""
    print("🔍 GPU Monitoring (Ctrl+C to stop)")
    print("=" * 50)
    
    try:
        while True:
            timestamp = time.strftime("%H:%M:%S")
            gpu_info = get_gpu_info()
            print(f"[{timestamp}] {gpu_info}")
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")

if __name__ == "__main__":
    main()
