# Meta Tensor Bug Reproducers

This directory contains test scripts documenting a bug where `at::mm()` hangs when called on meta tensors with autograd enabled in certain build configurations.

## Quick Start

```bash
# Test F.linear with meta tensors (actual use case)
python repro/reproduce_linear_meta_hang.py

# Test various meta tensor creation methods
python repro/reproduce_with_cuda_meta.py
```

## Files

- **`reproduce_linear_meta_hang.py`** - Tests `F.linear` with meta tensors (matches actual use case where hang was observed)
- **`reproduce_with_cuda_meta.py`** - Tests various meta tensor creation methods and dtypes
- **`REPRODUCE_META_HANG_BUG.md`** - Complete bug documentation and analysis
- **`README.md`** - This file

## Expected Behavior

In the environment where the bug was originally observed:
- Tests with 3D × 2D matmul would hang indefinitely
- The hang occurred in the C++ dispatcher before reaching any implementation
- Dispatch keys showed: `DispatchKeySet(Meta, ADInplaceOrView, AutogradMeta)`

In a vanilla PyTorch build:
- All tests likely pass (no hang)
- The workaround in `LinearAlgebra.cpp` prevents the issue

## The Bug

**Location:** C++ dispatcher, when calling `at::mm()` on meta tensors

**Trigger:** The should_fold optimization path in `_matmul_impl` when:
- Input tensors are on meta device
- Autograd is enabled (AutogradMeta dispatch key present)
- Operation is 3D × 2D or similar (triggers folding)

**Workaround:** Manually create output tensor for meta device:
```cpp
if (tensor.device().is_meta()) {
  result = at::empty({M, N}, tensor.options());
}
```

## Debug Output

The original debug output showed:
```
[DEBUG _matmul_impl] t1_folded.key_set() = DispatchKeySet(Meta, ADInplaceOrView, AutogradMeta)
[DEBUG _matmul_impl] t2->key_set() = DispatchKeySet(Meta, ADInplaceOrView, AutogradMeta)
[DEBUG _matmul_impl] Now calling at::mm...
<HANG - No further output>
```

This confirmed:
1. Both tensors were meta tensors
2. AutogradMeta key was present
3. Python dispatch was NOT active
4. Hang occurred at the C++ dispatcher level

## Related Code

- `aten/src/ATen/native/LinearAlgebra.cpp` - Contains workaround
- `aten/src/ATen/native/Linear.cpp` - Debug prints in F.linear
- `torch/_meta_registrations.py` - Python meta implementations
- `torch/_subclasses/fake_tensor.py` - FakeTensor dispatch

## More Information

See `REPRODUCE_META_HANG_BUG.md` for complete details.

