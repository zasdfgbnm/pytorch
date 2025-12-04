# PyTorch Meta Tensor MM Hang Bug - Reproducer

## Bug Description

When calling `torch.matmul()` or `at::mm()` on plain meta tensors with autograd enabled, under certain conditions the program can hang indefinitely in the C++ dispatcher. The hang occurs:

- **Before** any implementation is reached (neither C++ nor Python meta kernels)
- **In the dispatcher** itself, somewhere between the function call and kernel dispatch
- **Only when** the should_fold optimization path is triggered (e.g., 3D × 2D matmul)

**Note:** This bug was observed in a specific build/environment. The reproducers in this directory may not trigger the hang in a vanilla PyTorch build, but they document the issue and test various scenarios.

### Dispatch Keys Involved

```
DispatchKeySet(Meta, ADInplaceOrView, AutogradMeta)
```

The presence of `AutogradMeta` appears to be the key issue.

## Reproducer Files

### 1. `reproduce_linear_meta_hang.py`
Tests `F.linear` with meta tensors - the actual use case where the hang was observed.

**Usage:**
```bash
python repro/reproduce_linear_meta_hang.py
```

**Tests:**
- F.linear with 2D, 3D, and 4D inputs on meta device
- With and without torch.no_grad()
- Matches the original scenario that triggered the bug

**Note:** This reproducer includes timeout handling. In the environment where the bug was observed, Test 2 (3D input) would hang. With the C++ workaround in place, all tests pass.

### 2. `reproduce_with_cuda_meta.py`
Tests various ways of creating and using meta tensors.

**Usage:**
```bash
python repro/reproduce_with_cuda_meta.py
```

**Tests:**
- Plain meta tensors
- Meta tensors with CUDA device context
- BFloat16 dtype (matches debug output from original bug)
- Meta tensors with requires_grad=True
- Different tensor creation methods (torch.randn, torch.empty, etc.)

**Purpose:** Helps identify which specific combination of meta tensor attributes triggers the bug.

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

Run the reproducers to test for the bug:

```bash
# Test F.linear with meta tensors (actual use case where hang was observed)
python repro/reproduce_linear_meta_hang.py

# Test different meta tensor creation methods and configurations
python repro/reproduce_with_cuda_meta.py
```

## Why Tests May Not Hang

The tests in this directory **do not hang** in vanilla PyTorch builds. This is because:

1. **Workaround is active**: The C++ workaround in `LinearAlgebra.cpp` bypasses `at::mm()` for meta tensors
2. **Environment-specific**: The original bug only occurred in a specific build configuration
3. **May be fixed**: The underlying dispatcher issue may have been resolved

The workaround in `aten/src/ATen/native/LinearAlgebra.cpp` (around line 2226):
```cpp
if (t1_folded.device().is_meta() && t2->device().is_meta()) {
  // Manually create output - bypasses the hang
  auto M = t1_folded.size(0);
  auto N = t2->size(1);
  mm_result = at::empty({M, N}, t1_folded.options());
}
```

### Purpose of These Reproducers

These scripts serve as:
- **Documentation** of the bug that was observed
- **Test cases** to verify the workaround works correctly
- **Regression tests** if the workaround is removed
- **Reference** for understanding the code paths involved

To actually trigger the hang, you would need to:
1. Remove the workaround code above
2. Rebuild PyTorch
3. Run the tests in the specific environment where the bug occurs

## Reporting

When reporting this bug to PyTorch:

1. Include output from the reproducers (especially `reproduce_linear_meta_hang.py`)
2. Specify PyTorch version: `python -c "import torch; print(torch.__version__)"`
3. Include the debug output showing dispatch keys:
   ```
   DispatchKeySet(Meta, ADInplaceOrView, AutogradMeta)
   ```
4. Mention it's a C++ dispatcher hang, not a Python issue
5. Note the workaround works (manually creating meta tensor with correct shape)
6. Include information about your build configuration if it differs from vanilla PyTorch

