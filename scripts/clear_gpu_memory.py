"""
Clear GPU memory and show current usage.

Run this before fine-tuning if you encounter OOM errors.
"""

import gc

import torch

print("Clearing GPU memory...")

# Clear Python garbage
gc.collect()

# Clear PyTorch cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Show memory stats
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")

        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3

        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved: {reserved:.2f} GB")
        print(f"  Total: {total:.2f} GB")
        print(f"  Free: {total - allocated:.2f} GB")

    print("\n✓ GPU memory cleared")
else:
    print("No GPU available")
