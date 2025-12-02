# Debug Prints Added to PyTorch

This document lists all the debug prints added throughout PyTorch to trace the dispatcher path for your complex code.

## Files Modified

### 1. `aten/src/ATen/core/dispatch/OperatorEntry.h`

**Function:** `lookup(DispatchKeySet ks)`

Prints added:
- Entry to lookup with operator name and keyset
- Highest priority dispatch key
- Dispatch table index
- Kernel validity checks
- Before returning kernel

Look for: `[DEBUG OperatorEntry::lookup]`

### 2. `aten/src/ATen/core/dispatch/OperatorEntry.cpp`

**Function:** `computeDispatchTableEntryWithDebug()`

Prints added at:
- Function entry with operator name and dispatch key
- Path 1: Direct registration
- Path 2.2: CompositeExplicitAutograd  
- Path 2.3: CompositeImplicitAutograd
- Path 3: Backend fallback
- Path 4: Missing kernel

Look for: `[DEBUG OperatorEntry::computeDispatchTableEntryWithDebug]` and `[DEBUG computeDispatchTableEntry]`

### 3. `aten/src/ATen/core/dispatch/Dispatcher.h`

**Function:** `redispatchBoxed()`

Prints added:
- Entry with operator name and dispatch keyset
- Before calling entry.lookup()
- After lookup, before kernel.callBoxed()
- After kernel.callBoxed() returns

Look for: `[DEBUG Dispatcher::redispatchBoxed]`

### 4. `aten/src/ATen/core/dispatch/Dispatcher.cpp`

**Function:** `_print_dispatch_trace()`

Enhanced to show full dispatch key set for mm operations.

Look for: `[DEBUG DISPATCHER]`

### 5. `aten/src/ATen/native/LinearAlgebra.cpp`

Already has extensive prints:
- In `_matmul_impl` before calling at::mm
- In `TORCH_META_FUNC(mm)` when meta kernel is entered
- In `TORCH_IMPL_FUNC(mm_out_cpu)` when CPU kernel is entered

Look for: `[DEBUG _matmul_impl]`, `[DEBUG TORCH_META_FUNC(mm)]`, `[DEBUG TORCH_IMPL_FUNC(mm_out_cpu)]`

## Complete Call Stack Trace

When you run your code, you should see (in order):

### 1. Entry from your code
```
[DEBUG _matmul_impl] About to call at::mm WITHOUT NoGradGuard...
[DEBUG _matmul_impl] t1_folded.key_set() = DispatchKeySet(Meta, ADInplaceOrView, AutogradMeta)
[DEBUG _matmul_impl] Inside NoGradGuard, calling at::mm...
```

### 2. Dispatcher::call() entry (unboxed path)
```
[DEBUG Dispatcher::call] op=aten::mm
[DEBUG Dispatcher::call] dispatchKeySet=DispatchKeySet(Meta, AutogradMeta)  
[DEBUG Dispatcher::call] About to call lookup()
```

### 3. First lookup (for AutogradMeta)
```
[DEBUG OperatorEntry::lookup] op=aten::mm, keyset=DispatchKeySet(Meta, AutogradMeta)
[DEBUG OperatorEntry::lookup] highest priority key=AutogradMeta
[DEBUG OperatorEntry::lookup] dispatch table index=112
[DEBUG OperatorEntry::lookup] kernel.isValid=1, isValidUnboxed=1
[DEBUG OperatorEntry::lookup] Returning kernel, about to call it
```

### 4. Dispatcher calls kernel (unboxed)
```
[DEBUG Dispatcher::call] lookup() returned
[DEBUG Dispatcher::call] About to call kernel.call<>() [fast path]
```

### 5. Autograd kernel redispatches (removes AutogradMeta key)
This calls `Dispatcher::redispatchBoxed()` or similar, which triggers:

### 6. Second lookup (for Meta)
```
[DEBUG OperatorEntry::lookup] op=aten::mm, keyset=DispatchKeySet(Meta)
[DEBUG OperatorEntry::lookup] highest priority key=Meta
[DEBUG OperatorEntry::lookup] dispatch table index=16
[DEBUG OperatorEntry::lookup] kernel.isValid=1, isValidUnboxed=0  ← BOXED ONLY
[DEBUG OperatorEntry::lookup] Returning kernel, about to call it
```

### 7. Kernel invocation (boxed path)
```
[DEBUG Dispatcher::callBoxed] About to call kernel.callBoxed() [fast path]
[DEBUG KernelFunction::callBoxed] About to call boxed_kernel_func_.callBoxed()
[DEBUG BoxedKernel::callBoxed] About to invoke (*boxed_kernel_func_)(...)
```

### 8. Meta kernel execution
```
[DEBUG TORCH_META_FUNC(mm)] ========== ENTERED MM META FUNCTION ==========
[DEBUG TORCH_META_FUNC(mm)] self shape: ...
[DEBUG TORCH_META_FUNC(mm)] Meta function complete
```

### 9. Return path
```
[DEBUG BoxedKernel::callBoxed] (*boxed_kernel_func_)() RETURNED!
[DEBUG KernelFunction::callBoxed] boxed_kernel_func_.callBoxed() RETURNED
[DEBUG Dispatcher::callBoxed] kernel.callBoxed() RETURNED [fast path]
[DEBUG _matmul_impl] at::mm RETURNED!
```

## If It Hangs

### Where to Look

Compare your output to the expected sequence above. The **last print** before the hang tells you exactly where it's stuck:

| Last Print | What It Means |
|------------|---------------|
| `About to call at::mm...` | Hang is in entering dispatcher |
| `About to call entry.lookup()` | Hang is in lookup function |
| `dispatch table index=N` | Hang is in kernel validity check |
| `about to call kernel.callBoxed()` | Hang is in kernel invocation |
| `kernel.callBoxed()` printed but no RETURNED | Hang is **inside the kernel itself** |
| No `TORCH_META_FUNC(mm)` after callBoxed | Meta kernel was never entered - dispatcher routing issue |

### Most Likely Scenarios

Based on the original bug report:

1. **Hang after "about to call kernel.callBoxed()"**
   - The dispatcher successfully found the kernel
   - But the kernel.callBoxed() call hangs
   - This suggests an issue in the boxing/unboxing layer or in autograd hooks

2. **Hang in lookup()**
   - The dispatch table lookup itself is stuck
   - Could be an infinite loop in dispatch key resolution

3. **Kernel called but never entered**
   - callBoxed() was called
   - But TORCH_META_FUNC(mm) was never reached
   - Suggests the kernel pointer is wrong or there's an intermediate layer

## How to Use

### Run Your Code

```bash
# Your complex code that triggers the hang
python your_script.py 2>&1 | tee debug_output.txt
```

### Optionally Enable Built-in Tracing

```bash
export TORCH_SHOW_DISPATCH_TRACE=1
python your_script.py 2>&1 | tee debug_output.txt
```

### Analyze Output

1. Find the last `[DEBUG ...]` print before the hang
2. Compare to the expected sequence above
3. This tells you exactly where in the dispatcher the hang occurs

### If You Need More Detail

You can add even more prints by:
- Adding prints in `KernelFunction::callBoxed()` implementation
- Adding prints in autograd hooks
- Adding prints in Python dispatch interceptors

## Additional Debugging

### With GDB

```bash
gdb --args python your_script.py
(gdb) run
# Wait for hang, then Ctrl+C
(gdb) bt  # Get C++ backtrace
(gdb) thread apply all bt  # All threads
```

### Check for Deadlock

```bash
# In another terminal while hung:
kill -QUIT <python_pid>  # Sends SIGQUIT, dumps Python traceback
```

## Notes

- All mm-related operations will print debug info
- Prints go to stderr for better visibility
- Prints are flushed immediately to ensure they appear before any hang
- The "is_mm" check looks for "mm" in the operator name, so it catches:
  - `aten::mm`
  - `aten::addmm`
  - `aten::bmm`
  - etc.

## Cleanup

After debugging, you may want to remove or disable these prints for performance. They are all wrapped in:
```cpp
if (is_mm) {
  std::cerr << "[DEBUG ...] ..." << std::endl;
}
```

So they only affect mm-related operations.

