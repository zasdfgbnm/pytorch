"""
Minimal reproducer for PyTorch meta tensor mm hang bug.

This script demonstrates that calling matmul on meta tensors with autograd
enabled causes the program to hang in the C++ dispatcher.

The hang occurs in at::mm() when:
1. Both tensors are on meta device
2. Tensors have autograd dispatch keys (Meta, ADInplaceOrView, AutogradMeta)
3. The operation goes through the should_fold path in _matmul_impl

Expected behavior: Should compute output shape and return meta tensor
Actual behavior: Hangs indefinitely in C++ dispatcher before reaching any implementation
"""

import torch
import sys

print("=" * 80)
print("PyTorch Meta Tensor MM Hang Bug Reproducer")
print("=" * 80)
print()

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print()

# Test 1: Simple 2D x 2D matmul (should work)
print("Test 1: Simple 2D x 2D matmul on meta device")
print("-" * 60)
try:
    with torch.device('meta'):
        a = torch.randn(4, 8)
        b = torch.randn(8, 16)
    
    print(f"a.shape: {a.shape}, a.device: {a.device}")
    print(f"b.shape: {b.shape}, b.device: {b.device}")
    print(f"a.key_set: {a._debug_dispatch_key_set()}" if hasattr(a, '_debug_dispatch_key_set') else "")
    print("Calling torch.matmul(a, b)...")
    
    result = torch.matmul(a, b)
    print(f"✓ Success! result.shape: {result.shape}")
except Exception as e:
    print(f"✗ Failed: {e}")
print()

# Test 2: 3D x 2D matmul (triggers should_fold path - this hangs!)
print("Test 2: 3D x 2D matmul on meta device (triggers should_fold path)")
print("-" * 60)
print("WARNING: This will hang! Press Ctrl+C to interrupt after 5 seconds")
print()

try:
    with torch.device('meta'):
        a = torch.randn(2, 4, 16)  # 3D tensor
        b = torch.randn(16, 16)     # 2D tensor
    
    print(f"a.shape: {a.shape}, a.device: {a.device}")
    print(f"b.shape: {b.shape}, b.device: {b.device}")
    print(f"GradMode.is_enabled: {torch.is_grad_enabled()}")
    
    # This triggers the should_fold optimization in _matmul_impl
    # which calls mm() on the folded tensors
    print("Calling torch.matmul(a, b) - this will hang...")
    sys.stdout.flush()
    
    result = torch.matmul(a, b)
    
    # If we get here, the bug is fixed!
    print(f"✓ Success! result.shape: {result.shape}")
except KeyboardInterrupt:
    print("\n✗ Interrupted - program was hanging as expected")
    sys.exit(1)
except Exception as e:
    print(f"✗ Failed with exception: {e}")
    import traceback
    traceback.print_exc()
print()

# Test 3: Same as test 2 but with grad disabled
print("Test 3: 3D x 2D matmul with torch.no_grad()")
print("-" * 60)
try:
    with torch.device('meta'):
        a = torch.randn(2, 4, 16)
        b = torch.randn(16, 16)
    
    print(f"a.shape: {a.shape}, a.device: {a.device}")
    print(f"b.shape: {b.shape}, b.device: {b.device}")
    
    with torch.no_grad():
        print(f"GradMode.is_enabled: {torch.is_grad_enabled()}")
        print("Calling torch.matmul(a, b) with no_grad...")
        sys.stdout.flush()
        
        result = torch.matmul(a, b)
        print(f"✓ Success! result.shape: {result.shape}")
except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
print()

print("=" * 80)
print("Bug Summary:")
print("=" * 80)
print("""
The hang occurs when:
1. Tensors are on meta device
2. Autograd is enabled (default)
3. The operation triggers the should_fold optimization path
   (e.g., 3D x 2D, 2D x 3D matmul)

The hang happens in the C++ dispatcher at at::mm() call before reaching
any implementation (neither C++ TORCH_META_FUNC(mm) nor Python meta_mm).

Workaround:
- Use torch.no_grad() context
- Or manually create output tensor with correct shape for meta tensors
""")

