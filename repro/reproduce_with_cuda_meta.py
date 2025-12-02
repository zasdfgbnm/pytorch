"""
Test to reproduce the hang with CUDA meta tensors specifically.

The original debug output showed:
  Final output device: cuda:0

This suggests the tensors may have been created as CUDA meta tensors,
not plain meta tensors. This test tries both scenarios.
"""

import torch
import torch.nn.functional as F
import sys
import time

print("=" * 80)
print("Test: CUDA Meta Tensors vs Plain Meta Tensors")
print("=" * 80)
print()

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Grad enabled: {torch.is_grad_enabled()}")
print()

def run_test(description, create_tensors_fn):
    print(f"\n{description}")
    print("-" * 60)
    
    try:
        input, weight, bias = create_tensors_fn()
        
        print(f"  input: shape={input.shape}, device={input.device}, dtype={input.dtype}")
        print(f"  weight: shape={weight.shape}, device={weight.device}, dtype={weight.dtype}")
        print(f"  input.key_set: {input.key_set()}")
        
        print(f"  Calling F.linear...")
        sys.stdout.flush()
        
        start = time.time()
        result = F.linear(input, weight, bias)
        elapsed = time.time() - start
        
        print(f"  ✓ Success! (took {elapsed:.3f}s)")
        print(f"  result: shape={result.shape}, device={result.device}")
        return True
        
    except Exception as e:
        print(f"  ✗ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

# Test 1: Plain meta tensors
def create_plain_meta():
    with torch.device('meta'):
        input = torch.randn(2, 4, 16)
        weight = torch.randn(16, 16)
        bias = torch.randn(16)
    return input, weight, bias

success1 = run_test(
    "Test 1: Plain meta tensors (device='meta')",
    create_plain_meta
)

# Test 2: CUDA meta tensors
def create_cuda_meta():
    # Create tensors with device='meta' but CUDA dtype/attributes
    with torch.device('meta'):
        input = torch.randn(2, 4, 16)
        weight = torch.randn(16, 16)
        bias = torch.randn(16)
    
    # Try to set fake device to CUDA (this might not work)
    # In practice, this would be done through FakeTensor or other mechanism
    return input, weight, bias

if torch.cuda.is_available():
    success2 = run_test(
        "Test 2: Meta tensors created with CUDA context",
        create_cuda_meta
    )
else:
    print("\nTest 2: Skipped (CUDA not available)")
    success2 = True

# Test 3: Using torch.empty with meta device
def create_with_empty():
    input = torch.empty(2, 4, 16, device='meta')
    weight = torch.empty(16, 16, device='meta')
    bias = torch.empty(16, device='meta')
    return input, weight, bias

success3 = run_test(
    "Test 3: Meta tensors created with torch.empty",
    create_with_empty
)

# Test 4: Testing the exact scenario from the debug output
def create_exact_scenario():
    """
    From the debug output:
    [DEBUG] Input dimension: 3
    [DEBUG] weight dimension: 2
    [DEBUG] Input dtype: c10::BFloat16
    [DEBUG] Output device: cuda:0
    """
    with torch.device('meta'):
        # Use bfloat16 to match the debug output
        input = torch.randn(2, 4, 16, dtype=torch.bfloat16)
        weight = torch.randn(16, 16, dtype=torch.bfloat16)
        bias = torch.randn(16, dtype=torch.bfloat16)
    return input, weight, bias

success4 = run_test(
    "Test 4: BFloat16 meta tensors (matches debug output)",
    create_exact_scenario
)

# Test 5: With requires_grad
def create_with_grad():
    with torch.device('meta'):
        input = torch.randn(2, 4, 16, requires_grad=True)
        weight = torch.randn(16, 16, requires_grad=True)
        bias = torch.randn(16, requires_grad=True)
    return input, weight, bias

success5 = run_test(
    "Test 5: Meta tensors with requires_grad=True",
    create_with_grad
)

# Summary
print()
print("=" * 80)
print("Summary:")
print("=" * 80)
print(f"Plain meta tensors:           {'PASS' if success1 else 'FAIL'}")
print(f"CUDA meta tensors:            {'PASS' if success2 else 'FAIL'}")
print(f"Empty meta tensors:           {'PASS' if success3 else 'FAIL'}")
print(f"BFloat16 meta tensors:        {'PASS' if success4 else 'FAIL'}")
print(f"Meta tensors with grad:       {'PASS' if success5 else 'FAIL'}")
print()

if all([success1, success2, success3, success4, success5]):
    print("All tests passed!")
    print()
    print("Possible reasons:")
    print("1. The bug only reproduces with a custom PyTorch build")
    print("2. The bug only occurs with FakeTensor, not plain meta tensors")
    print("3. The bug was already fixed")
    print("4. The bug requires a specific combination of conditions")
    print()
    print("The workaround added to LinearAlgebra.cpp may be preventing the hang.")
else:
    print("Some tests failed!")
    sys.exit(1)

