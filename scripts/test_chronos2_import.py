"""
Test script to verify Chronos 2.0 imports and available models.
"""

import sys

print("Python version:", sys.version)
print("Testing chronos-forecasting 2.0 imports")

# Try different import methods
try:
    import chronos

    print("import chronos - SUCCESS")
    print(f"Version: {chronos.__version__ if hasattr(chronos, '__version__') else 'unknown'}")
    print(f"Available: {[x for x in dir(chronos) if not x.startswith('_')]}")
except ImportError as e:
    print(f"import chronos - FAILED: {e}")

try:
    from chronos import ChronosPipeline

    print("from chronos import ChronosPipeline - SUCCESS")
except ImportError as e:
    print(f"from chronos import ChronosPipeline - FAILED: {e}")

try:
    from chronos import BaseChronosPipeline

    print("from chronos import BaseChronosPipeline - SUCCESS")
except ImportError as e:
    print(f"from chronos import BaseChronosPipeline - FAILED: {e}")

# Check torch
try:
    import torch

    print(f"PyTorch {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("PyTorch not available")

print("Test completed")
