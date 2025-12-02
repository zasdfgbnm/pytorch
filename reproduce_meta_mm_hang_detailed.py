"""
Detailed reproducer for PyTorch meta tensor mm hang bug.

This version provides more diagnostic information about the tensors
and dispatch keys involved in the hang.
"""

import torch
import sys

def print_tensor_info(name, tensor):
    """Print detailed information about a tensor."""
    print(f"{name}:")
    print(f"  shape: {tensor.shape}")
    print(f"  device: {tensor.device}")
    print(f"  dtype: {tensor.dtype}")
    print(f"  requires_grad: {tensor.requires_grad}")
    print(f"  is_contiguous: {tensor.is_contiguous()}")
    
    # Try to get dispatch key information
    if hasattr(tensor, '_debug_dispatch_key_set'):
        print(f"  dispatch keys: {tensor._debug_dispatch_key_set()}")
    
    # Check if it has storage
    try:
        storage = tensor.untyped_storage()
        print(f"  has_storage: True (size: {storage.size()})")
    except:
        print(f"  has_storage: False or error")
    print()

print("=" * 80)
print("PyTorch Meta Tensor MM Hang - Detailed Reproducer")
print("=" * 80)
print()

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Grad mode enabled:", torch.is_grad_enabled())
print()

# The problematic case
print("Creating meta tensors that will trigger the hang:")
print("-" * 60)

with torch.device('meta'):
    # Create 3D tensor - this will trigger should_fold path
    tensor1 = torch.randn(2, 4, 16)  # Shape that causes folding to [8, 16]
    tensor2 = torch.randn(16, 16)    # 2D tensor
    
print_tensor_info("tensor1 (3D)", tensor1)
print_tensor_info("tensor2 (2D)", tensor2)

print("The matmul operation will:")
print("1. Detect that tensor1.dim() >= 3 and tensor2.dim() == 2")
print("2. Enter the should_fold optimization path in _matmul_impl")
print("3. Fold tensor1 from [2, 4, 16] to [8, 16]")
print("4. Call at::mm(folded_tensor1, tensor2)")
print("5. HANG in the C++ dispatcher before reaching any implementation")
print()

print("Expected output shape: [2, 4, 16]")
print()

# Add timeout mechanism
import signal
import time

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

print("Setting 5-second timeout...")
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(5)

try:
    print("Calling torch.matmul(tensor1, tensor2)...")
    sys.stdout.flush()
    
    start_time = time.time()
    result = torch.matmul(tensor1, tensor2)
    elapsed = time.time() - start_time
    
    signal.alarm(0)  # Cancel the alarm
    
    print(f"✓ SUCCESS! (took {elapsed:.3f}s)")
    print(f"result.shape: {result.shape}")
    print()
    print("The bug appears to be FIXED!")
    
except TimeoutException:
    signal.alarm(0)
    print()
    print("✗ TIMEOUT - Program hung as expected!")
    print()
    print("This confirms the bug:")
    print("  - at::mm() is hanging in the C++ dispatcher")
    print("  - The hang occurs before reaching any meta implementation")
    print("  - Dispatch keys involved: Meta, ADInplaceOrView, AutogradMeta")
    sys.exit(1)
    
except KeyboardInterrupt:
    signal.alarm(0)
    print()
    print("✗ INTERRUPTED - Program was hanging")
    sys.exit(1)
    
except Exception as e:
    signal.alarm(0)
    print(f"✗ Exception: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 80)
print("Additional Information:")
print("=" * 80)
print("""
Debug output from instrumented PyTorch build shows:

[DEBUG _matmul_impl] t1_folded.key_set() = DispatchKeySet(Meta, ADInplaceOrView, AutogradMeta)
[DEBUG _matmul_impl] t2->key_set() = DispatchKeySet(Meta, ADInplaceOrView, AutogradMeta)
[DEBUG _matmul_impl] t1_folded.has_storage() = 1
[DEBUG _matmul_impl] t2->has_storage() = 1
[DEBUG _matmul_impl] t1_folded.is_python_dispatch() = 0
[DEBUG _matmul_impl] Now calling at::mm...
<HANG OCCURS HERE>

The hang happens:
- After the at::mm() call is made
- Before TORCH_META_FUNC(mm) is entered (C++ meta kernel)
- Before meta_mm() is entered (Python meta registration)
- Somewhere in the C++ dispatcher itself

Possible causes:
1. Autograd dispatcher getting stuck with meta tensors
2. View/aliasing tracking causing infinite loop
3. Structured kernel dispatch system issue with meta + autograd keys

Workaround in C++ code:
  if (tensor.device().is_meta()) {
    // Manually create output with correct shape
    result = at::empty({M, N}, tensor.options());
  } else {
    result = at::mm(tensor1, tensor2);
  }
""")

