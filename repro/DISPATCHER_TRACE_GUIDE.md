# Dispatcher Tracing Guide

## How to Trace the Dispatch Path

### Option 1: Built-in Dispatcher Trace

PyTorch has a built-in dispatcher tracing facility:

```bash
# Enable dispatcher tracing
export TORCH_SHOW_DISPATCH_TRACE=1

# Run your test
python repro/test_with_dispatch_trace.py
```

Or use the provided script:

```bash
chmod +x repro/test_hang_with_full_trace.sh
./repro/test_hang_with_full_trace.sh
```

### Option 2: C++ Debug Prints

The codebase has been instrumented with debug prints at key points:

1. **`_matmul_impl`** in `aten/src/ATen/native/LinearAlgebra.cpp`
   - Shows when `at::mm` is about to be called
   - Shows tensor properties (shape, device, dispatch keys)

2. **`TORCH_META_FUNC(mm)`** in `aten/src/ATen/native/LinearAlgebra.cpp`  
   - Shows when the meta kernel is entered
   - **If you don't see this, the dispatcher never reached it**

3. **`TORCH_IMPL_FUNC(mm_out_cpu)`** in `aten/src/ATen/native/LinearAlgebra.cpp`
   - Shows CPU implementation entry

4. **Dispatcher** in `aten/src/ATen/core/dispatch/Dispatcher.cpp`
   - Enhanced `_print_dispatch_trace` with extra info for mm ops

## Expected Dispatch Path (Normal Case)

For meta tensors, the dispatch should go:

```
at::mm(tensor1, tensor2)
  ↓
[Dispatcher] Looking at dispatch keys: Meta, ADInplaceOrView, AutogradMeta
  ↓
[Autograd layer] Since AutogradMeta is present
  ↓
[Meta dispatch] Dispatch to Meta kernel
  ↓
TORCH_META_FUNC(mm) - Compute output shape
  ↓
Return meta tensor with correct shape
```

## What to Look For If It Hangs

### Last Prints Before Hang

```
[DEBUG _matmul_impl] About to call at::mm WITHOUT NoGradGuard...
[DEBUG _matmul_impl STDERR] === CALLING at::mm NOW ===
<HANG - No further output>
```

### What's Missing

If it hangs, you will **NOT** see:
- `[DEBUG TORCH_META_FUNC(mm)] ========== ENTERED MM META FUNCTION ==========`
- Any prints from Python `meta_mm()`  
- Dispatcher trace showing Meta kernel being called

This confirms the hang is **in the dispatcher** before reaching any implementation.

## Dispatch Keys to Watch For

From the original bug, the tensors had:

```
DispatchKeySet(Meta, ADInplaceOrView, AutogradMeta)
```

Key observations:
- **Meta**: Tensor is on meta device
- **AutogradMeta**: Autograd tracking is active
- **ADInplaceOrView**: Aliasing/view tracking
- **NO Python**: Python dispatch is not involved

## Debugging the Dispatcher

### Check Current Dispatch Keys

In C++ debug prints, look for:
```
[DEBUG _matmul_impl] t1_folded.key_set() = DispatchKeySet(...)
```

### Check Autograd Status

```
[DEBUG _matmul_impl] GradMode::is_enabled() = 1
```

If `GradMode::is_enabled() = 1`, autograd is active and may be involved in the hang.

## Code Changes Made

### 1. Removed NoGradGuard

**Before** (was preventing hang):
```cpp
{
  at::NoGradGuard no_grad;
  mm_result = at::mm(t1_folded, *t2);
}
```

**After** (allows hang to reproduce):
```cpp
mm_result = at::mm(t1_folded, *t2);
```

### 2. Removed Workaround

**Before** (bypassed at::mm):
```cpp
if (t1_folded.device().is_meta() && t2->device().is_meta()) {
  mm_result = at::empty({M, N}, t1_folded.options());
}
```

**After** (calls at::mm directly):
```cpp
mm_result = at::mm(t1_folded, *t2);
```

## Running the Tests

### Test 1: With Dispatcher Trace

```bash
export TORCH_SHOW_DISPATCH_TRACE=1
python repro/test_with_dispatch_trace.py
```

Watch for dispatcher output showing the routing of the `mm` operation.

### Test 2: With Timeout

```bash
# Will timeout if hangs
python repro/reproduce_linear_meta_hang.py
```

### Test 3: Direct Test

```bash
python3 << EOF
import torch
import torch.nn.functional as F

with torch.device('meta'):
    input = torch.randn(2, 4, 16)
    weight = torch.randn(16, 16)
    bias = torch.randn(16)

result = F.linear(input, weight, bias)
print(f"Result: {result.shape}")
EOF
```

## Interpreting Results

### If It Hangs

The bug is reproduced! The hang occurs in the C++ dispatcher when:
- Meta tensors with AutogradMeta dispatch key
- Calling `at::mm` through the should_fold optimization
- Before reaching any kernel implementation

### If It Doesn't Hang

Possible reasons:
1. Environment difference (Python version, build configuration, etc.)
2. PyTorch version has a fix
3. The specific combination of conditions isn't met

### If You See TORCH_META_FUNC Output

The dispatcher successfully routed to the meta kernel - no hang!

## Next Steps

1. **If it hangs**: Attach a debugger to see the exact stack trace
2. **Compare environments**: Check PyTorch version, build flags, dependencies  
3. **Check dispatcher state**: Add more debug prints to see where it loops/blocks
4. **Test with GDB**: Run under debugger and interrupt when it hangs

## GDB Debugging

```bash
gdb python
(gdb) set env TORCH_SHOW_DISPATCH_TRACE=1
(gdb) run repro/test_with_dispatch_trace.py
# Wait for hang, then Ctrl+C
(gdb) bt  # Get backtrace
(gdb) info threads  # See all threads
```

This will show exactly where in the code the hang occurs.

