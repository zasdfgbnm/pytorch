# PyTorch Meta Tensor MM Hang Bug - Reproducer

## Bug Description

When calling `torch.matmul()` or `at::mm()` on plain meta tensors with autograd enabled, the program hangs indefinitely in the C++ dispatcher. The hang occurs:

- **Before** any implementation is reached (neither C++ nor Python meta kernels)
- **In the dispatcher** itself, somewhere between the function call and kernel dispatch
- **Only when** the should_fold optimization path is triggered (e.g., 3D × 2D matmul)

### Dispatch Keys Involved

```
DispatchKeySet(Meta, ADInplaceOrView, AutogradMeta)
```

The presence of `AutogradMeta` appears to be the key issue.

## Reproducer Files

### 1. `reproduce_meta_mm_hang.py`
Simple minimal reproducer that demonstrates the hang.

**Usage:**
```bash
python reproduce_meta_mm_hang.py
```

**Expected behavior:** Test 2 will hang. Press Ctrl+C after a few seconds.

### 2. `reproduce_meta_mm_hang_detailed.py`
Detailed version with more diagnostic information and automatic timeout.

**Usage:**
```bash
python reproduce_meta_mm_hang_detailed.py
```

**Expected behavior:** Will timeout after 5 seconds and print diagnostic info.

### 3. `test_meta_vs_fake_tensor.py`
Comparative test to check different scenarios.

**Usage:**
```bash
python test_meta_vs_fake_tensor.py
```

**Tests:**
- Plain meta tensors (3D × 2D) - **HANGS**
- FakeTensor mode (3D × 2D) - May work if FakeTensor has workarounds
- Plain meta + no_grad - May work if autograd is the issue
- Plain meta (2D × 2D) - Works (doesn't trigger folding)

## How to Reproduce

1. Use plain meta tensors (not FakeTensor):
   ```python
   with torch.device('meta'):
       a = torch.randn(2, 4, 16)  # 3D
       b = torch.randn(16, 16)     # 2D
   ```

2. Call matmul with autograd enabled:
   ```python
   result = torch.matmul(a, b)  # HANGS HERE
   ```

3. The hang occurs in the C++ dispatcher at the `at::mm()` call.

## Debug Output

With an instrumented PyTorch build, the last output before hang:

```
[DEBUG _matmul_impl] Calling at::mm(t1_folded, *t2) instead...
[DEBUG _matmul_impl] t1_folded.key_set() = DispatchKeySet(Meta, ADInplaceOrView, AutogradMeta)
[DEBUG _matmul_impl] t2->key_set() = DispatchKeySet(Meta, ADInplaceOrView, AutogradMeta)
[DEBUG _matmul_impl] t1_folded.has_storage() = 1
[DEBUG _matmul_impl] t2->has_storage() = 1
[DEBUG _matmul_impl] t1_folded.is_python_dispatch() = 0
[DEBUG _matmul_impl] t2->is_python_dispatch() = 0
[DEBUG _matmul_impl] Now calling at::mm...
<HANG OCCURS HERE - NO FURTHER OUTPUT>
```

## Workarounds

### C++ Workaround
In `aten/src/ATen/native/LinearAlgebra.cpp`:

```cpp
if (t1_folded.device().is_meta() && t2->device().is_meta()) {
  // Manually create output with correct shape
  auto M = t1_folded.size(0);
  auto N = t2->size(1);
  mm_result = at::empty({M, N}, t1_folded.options());
} else {
  mm_result = at::mm(t1_folded, *t2);
}
```

### Python Workaround
Use `torch.no_grad()` context:

```python
with torch.no_grad():
    result = torch.matmul(a, b)  # May work
```

Or use FakeTensor instead of plain meta tensors.

## Root Cause Analysis

The hang likely occurs in:

1. **Autograd dispatcher** - The `AutogradMeta` key causes autograd dispatch layer to be involved
2. **View/aliasing tracking** - Meta tensors with autograd might have issues in aliasing checks
3. **Structured kernel dispatch** - The `structured_delegate: mm.out` might not handle meta+autograd correctly

The fact that:
- 2D × 2D works (no folding)
- Python meta registration is not being called
- C++ TORCH_META_FUNC(mm) is not being called
- Hang is before any implementation

...suggests the issue is in the **dispatcher routing logic** itself, not in any specific kernel implementation.

## Environment

- PyTorch: Any recent version (bug exists in current main branch)
- Platform: Linux/macOS/Windows
- Device: Any (bug is device-independent)

## Expected Behavior

The operation should:
1. Dispatch to the meta kernel
2. Compute output shape: `[2, 4, 16]`
3. Return meta tensor with correct shape
4. Complete in < 1ms

## Actual Behavior

The operation:
1. Enters the C++ dispatcher
2. Never reaches any kernel implementation
3. Hangs indefinitely
4. Requires Ctrl+C to terminate

## Related Code Paths

- `aten/src/ATen/native/LinearAlgebra.cpp:_matmul_impl()` - Where the hang occurs
- `aten/src/ATen/native/native_functions.yaml` - mm registration with `structured_delegate`
- Autograd dispatcher layer (C++)
- View/aliasing tracking system

## Testing

Run the reproducers to confirm the bug still exists:

```bash
# Quick test (will hang - press Ctrl+C)
python reproduce_meta_mm_hang.py

# Test with timeout
python reproduce_meta_mm_hang_detailed.py

# Comprehensive test suite
python test_meta_vs_fake_tensor.py
```

## Reporting

When reporting this bug to PyTorch:

1. Include output from `reproduce_meta_mm_hang_detailed.py`
2. Specify PyTorch version: `python -c "import torch; print(torch.__version__)"`
3. Include the debug output showing dispatch keys
4. Mention it's a C++ dispatcher hang, not a Python issue
5. Note the workaround works (manually creating meta tensor)

