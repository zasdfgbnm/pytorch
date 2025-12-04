#!/bin/bash

# Script to test for the hang with full dispatcher tracing enabled

echo "========================================================================"
echo "Testing Meta Tensor MM Hang with Full Dispatcher Tracing"
echo "========================================================================"
echo ""
echo "This will show the complete dispatch path from at::mm to the kernel."
echo "If the bug reproduces, it will hang before reaching TORCH_META_FUNC(mm)."
echo ""
echo "Press Ctrl+C if it hangs for more than 10 seconds."
echo ""

# Enable dispatcher tracing
export TORCH_SHOW_DISPATCH_TRACE=1

# Run the test
python3 << 'EOF'
import torch
import torch.nn.functional as F

print("Creating meta tensors...")
with torch.device('meta'):
    input = torch.randn(2, 4, 16)  # 3D input
    weight = torch.randn(16, 16)    # 2D weight
    bias = torch.randn(16)

print(f"input: {input.shape}, device: {input.device}, requires_grad: {input.requires_grad}")
print(f"weight: {weight.shape}, device: {weight.device}, requires_grad: {weight.requires_grad}")
print("")
print("=" * 80)
print("Calling F.linear...")
print("=" * 80)

result = F.linear(input, weight, bias)

print("=" * 80)
print(f"SUCCESS! Result: {result.shape}")
print("=" * 80)
EOF

echo ""
echo "Test completed!"

