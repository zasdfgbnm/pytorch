"""
Reproducer for the actual bug: torch.nn.functional.linear with meta tensors.

This more closely matches the original scenario where the hang was observed.
The hang occurs inside F.linear when it calls matmul on meta tensors.
"""

import torch
import torch.nn.functional as F
import sys
import time
import signal

print("=" * 80)
print("PyTorch F.linear with Meta Tensors - Hang Reproducer")
print("=" * 80)
print()

print("PyTorch version:", torch.__version__)
print("Grad mode enabled:", torch.is_grad_enabled())
print()

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

def test_with_timeout(description, test_func, timeout=10):
    """Run a test with timeout."""
    print(f"\n{description}")
    print("-" * 60)
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    try:
        sys.stdout.flush()
        start = time.time()
        result = test_func()
        elapsed = time.time() - start
        signal.alarm(0)
        print(f"✓ Success! (took {elapsed:.3f}s)")
        if result is not None and hasattr(result, 'shape'):
            print(f"  Result shape: {result.shape}")
        return True
    except TimeoutException:
        signal.alarm(0)
        print(f"✗ TIMEOUT after {timeout}s - Operation hung!")
        return False
    except KeyboardInterrupt:
        signal.alarm(0)
        print(f"✗ Interrupted by user")
        return False
    except Exception as e:
        signal.alarm(0)
        print(f"✗ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

# Test 1: Simple case - 2D input
def test_linear_2d():
    with torch.device('meta'):
        input = torch.randn(4, 16)
        weight = torch.randn(16, 16)
        bias = torch.randn(16)
    
    print(f"  input: {input.shape}, device: {input.device}")
    print(f"  weight: {weight.shape}, device: {weight.device}")
    print(f"  bias: {bias.shape}, device: {bias.device}")
    
    return F.linear(input, weight, bias)

success1 = test_with_timeout(
    "Test 1: F.linear with 2D input on meta device",
    test_linear_2d
)

# Test 2: 3D input (batched) - this is what triggers the issue
def test_linear_3d():
    with torch.device('meta'):
        input = torch.randn(2, 4, 16)  # batch_size=2, seq_len=4, features=16
        weight = torch.randn(16, 16)
        bias = torch.randn(16)
    
    print(f"  input: {input.shape}, device: {input.device}")
    print(f"  weight: {weight.shape}, device: {weight.device}")
    print(f"  bias: {bias.shape}, device: {bias.device}")
    print(f"  This will call matmul(input, weight.t()) internally")
    print(f"  Which becomes: [2, 4, 16] @ [16, 16].t() = [2, 4, 16] @ [16, 16]")
    print(f"  This triggers the should_fold path!")
    
    return F.linear(input, weight, bias)

success2 = test_with_timeout(
    "Test 2: F.linear with 3D input (batched) - TRIGGERS HANG",
    test_linear_3d,
    timeout=10
)

# Test 3: With torch.no_grad()
def test_linear_3d_no_grad():
    with torch.device('meta'):
        input = torch.randn(2, 4, 16)
        weight = torch.randn(16, 16)
        bias = torch.randn(16)
    
    print(f"  input: {input.shape}, device: {input.device}")
    print(f"  Grad mode: {torch.is_grad_enabled()}")
    
    with torch.no_grad():
        return F.linear(input, weight, bias)

success3 = test_with_timeout(
    "Test 3: F.linear with 3D input + no_grad()",
    test_linear_3d_no_grad
)

# Test 4: 4D input (even more batching)
def test_linear_4d():
    with torch.device('meta'):
        input = torch.randn(2, 3, 4, 16)  # batch=2, dim1=3, seq_len=4, features=16
        weight = torch.randn(16, 16)
        bias = torch.randn(16)
    
    print(f"  input: {input.shape}, device: {input.device}")
    print(f"  weight: {weight.shape}, device: {weight.device}")
    
    return F.linear(input, weight, bias)

success4 = test_with_timeout(
    "Test 4: F.linear with 4D input (more batching)",
    test_linear_4d
)

# Summary
print()
print("=" * 80)
print("Summary:")
print("=" * 80)
print(f"F.linear 2D input:         {'PASS' if success1 else 'HANG/FAIL'}")
print(f"F.linear 3D input:         {'PASS' if success2 else 'HANG/FAIL'}")
print(f"F.linear 3D + no_grad:     {'PASS' if success3 else 'HANG/FAIL'}")
print(f"F.linear 4D input:         {'PASS' if success4 else 'HANG/FAIL'}")
print()

if not success2:
    print("Bug confirmed: F.linear with 3D meta tensor input hangs!")
    print()
    print("This is the actual bug reported:")
    print("- F.linear internally calls matmul(input, weight.t())")
    print("- With 3D input [2, 4, 16] and 2D weight [16, 16]")
    print("- matmul triggers should_fold optimization")
    print("- Folds input to [8, 16] and calls mm([8, 16], [16, 16])")
    print("- mm() hangs in C++ dispatcher with meta tensors + autograd keys")
    sys.exit(1)
elif not all([success1, success2, success3, success4]):
    print("Some tests failed or hung!")
    sys.exit(1)
else:
    print("All tests passed! Bug may be fixed or not reproducible in this environment.")
    sys.exit(0)

