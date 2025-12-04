"""
Test with PyTorch dispatcher tracing enabled.

This uses the built-in TORCH_SHOW_DISPATCH_TRACE environment variable
to show the full dispatch path.
"""

import os
# Enable dispatcher tracing BEFORE importing torch
os.environ['TORCH_SHOW_DISPATCH_TRACE'] = '1'

import torch
import torch.nn.functional as F

print("=" * 80)
print("Testing with TORCH_SHOW_DISPATCH_TRACE=1")
print("=" * 80)
print()

# Simple test that should show dispatch trace
print("Creating meta tensors...")
with torch.device('meta'):
    input = torch.randn(2, 4, 16)
    weight = torch.randn(16, 16)
    bias = torch.randn(16)

print(f"input: {input.shape}, device: {input.device}")
print(f"weight: {weight.shape}, device: {weight.device}")
print()

print("=" * 80)
print("Calling F.linear - watch for dispatch trace:")
print("=" * 80)
result = F.linear(input, weight, bias)

print()
print("=" * 80)
print(f"Success! result: {result.shape}")
print("=" * 80)

