#!/usr/bin/env python3
"""
Quick CUDA check for PyTorch.
Run: python check_cuda.py
"""

import sys

try:
    import torch
except Exception as e:
    print(f"PyTorch import failed: {e}")
    sys.exit(1)


def main() -> int:
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("CUDA not available. Using CPU.")
        return 0

    try:
        device_index = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_index)
        device_count = torch.cuda.device_count()
        compiled_cuda = getattr(torch.version, "cuda", "unknown")
        cudnn_version = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None

        print(f"CUDA device count: {device_count}")
        print(f"Current device index: {device_index}")
        print(f"Current device name: {device_name}")
        print(f"Compiled with CUDA: {compiled_cuda}")
        print(f"cuDNN enabled: {torch.backends.cudnn.enabled}, version: {cudnn_version}")

        # Sanity-check: run a tiny tensor op on GPU
        x = torch.randn(4, 4, device="cuda")
        y = x @ x.t()
        print(f"Tensor op on device: {y.device}, is_cuda: {y.is_cuda}")
        print("CUDA test: OK ✅")
        return 0
    except Exception as e:
        print(f"CUDA test failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())


