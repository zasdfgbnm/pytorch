# Complete Dispatch Path: at::mm to TORCH_META_FUNC(mm)

## Overview

This document explains the complete call path from user code calling `at::mm()` to the final `TORCH_META_FUNC(mm)` implementation, and where debug prints have been added.

## The Full Call Stack

```
1. User Python code: torch.mm(a, b)
   ↓
2. [C++] at::mm(tensor1, tensor2)                    ← Generated in Functions.h/Operators.cpp
   ↓
3. [C++] at::_ops::mm::call(tensor1, tensor2)        ← Generated in Operators.cpp
   ↓
4. [C++] TypedOperatorHandle::call()                 ← In Dispatcher.h
   ↓
5. [C++] Dispatcher::call()                          ← In Dispatcher.h
   ↓
6. [C++] OperatorEntry::lookup()                     ← In OperatorEntry.h
   ↓
7. [C++] KernelFunction::call()                      ← In KernelFunction_impl.h
   ↓
8. [C++] KernelFunction::callBoxed()                 ← In KernelFunction_impl.h
   ↓
9. [C++] BoxedKernel::callBoxed()                    ← In BoxedKernel_impl.h
   ↓
10. [C++] Lambda in BoxedKernel::makeFromFunctor     ← In BoxedKernel_impl.h
    ↓
11. [C++] KernelFunctor::operator()                  ← Generated structured kernel wrapper
    ↓
12. [C++] wrapper_Meta_mm()                          ← Generated in RegisterMeta.cpp
    ↓
13. [C++] structured_mm_meta::meta()                 ← Generated structured kernel class
    ↓
14. [C++] TORCH_META_FUNC(mm)                        ← In LinearAlgebra.cpp
```

## Debug Prints Added

### 1. **at::_ops::mm::call()** (NEW - in torchgen/gen.py)
   - **File**: `torchgen/gen.py` lines 670-694
   - **Prints**:
     ```
     [DEBUG at::_ops::mm::call] ===== ENTERED =====
     [DEBUG at::_ops::mm::call] About to get typed handle and call dispatcher
     [DEBUG at::_ops::mm::call] About to call op.call()
     [DEBUG at::_ops::mm::call] op.call() RETURNED
     [DEBUG at::_ops::mm::call] ===== EXITING =====
     ```
   - **Status**: ⚠️ **REQUIRES REGENERATION & REBUILD**

### 2. **Dispatcher::call()** (EXISTING)
   - **File**: `aten/src/ATen/core/dispatch/Dispatcher.h`
   - **Prints**: Shows operator name, dispatch key set, and before calling kernel

### 3. **Dispatcher::callBoxed()** (EXISTING)
   - **File**: `aten/src/ATen/core/dispatch/Dispatcher.h`
   - **Prints**: Shows operator name and dispatch key set for boxed calls

### 4. **OperatorEntry::lookup()** (EXISTING)
   - **File**: `aten/src/ATen/core/dispatch/OperatorEntry.h`
   - **Prints**: Shows operator name, keyset, highest priority key, dispatch table index

### 5. **KernelFunction::callBoxed()** (EXISTING)
   - **File**: `aten/src/ATen/core/boxing/KernelFunction_impl.h`
   - **Prints**: Shows dispatch keyset and before/after calling BoxedKernel

### 6. **BoxedKernel::callBoxed()** (EXISTING)
   - **File**: `aten/src/ATen/core/boxing/BoxedKernel_impl.h`
   - **Prints**: Shows function pointer address, functor pointer, stack info

### 7. **BoxedKernel::makeFromFunctor Lambda** (EXISTING)
   - **File**: `aten/src/ATen/core/boxing/BoxedKernel_impl.h`
   - **Prints**:
     ```
     [BoxedKernel LAMBDA] ===== ENTERED makeFromFunctor LAMBDA =====
     [BoxedKernel LAMBDA] >>>>> Calling functor operator() <<<<<
     [BoxedKernel LAMBDA] >>>>> Functor operator() RETURNED <<<<<
     ```
   - **Status**: ✅ **COMPILED** (User is seeing these prints)
   - **Hang Location**: Between "Calling functor operator()" and "RETURNED"

### 8. **wrapper_Meta_mm()** (NEW - in torchgen/dest/register_dispatch_key.py)
   - **File**: `torchgen/dest/register_dispatch_key.py` lines 1005-1015
   - **Prints**:
     ```
     [DEBUG structured_mm wrapper] ===== ENTERED WRAPPER =====
     [DEBUG structured_mm wrapper] Function: mm
     [DEBUG structured_mm wrapper] Creating op instance...
     [DEBUG structured_mm wrapper] ===== EXITING WRAPPER =====
     ```
   - **Status**: ⚠️ **REQUIRES REGENERATION & REBUILD**

### 9. **Before op.meta() call** (NEW - in torchgen/dest/register_dispatch_key.py)
   - **File**: `torchgen/dest/register_dispatch_key.py` line 878
   - **Prints**:
     ```
     [DEBUG structured_mm] About to call op.meta()
     ```
   - **Status**: ⚠️ **REQUIRES REGENERATION & REBUILD**

### 10. **After op.meta() call** (NEW - in torchgen/dest/register_dispatch_key.py)
   - **File**: `torchgen/dest/register_dispatch_key.py` line 909
   - **Prints**:
     ```
     [DEBUG structured_mm] op.meta() RETURNED
     ```
   - **Status**: ⚠️ **REQUIRES REGENERATION & REBUILD**

### 11. **TORCH_META_FUNC(mm)** (EXISTING)
   - **File**: `aten/src/ATen/native/LinearAlgebra.cpp` lines 203-233
   - **Prints**: Extensive shape/device/dtype information
   - **Status**: ✅ **COMPILED** (But user is NOT seeing these prints)

## Current Status

### What User Is Seeing:
```
[BoxedKernel LAMBDA] >>>>> Calling functor operator() <<<<<
```

### What User Is NOT Seeing:
- Anything after the above line
- Specifically, no `TORCH_META_FUNC(mm)` prints

### Conclusion:
The hang is occurring **inside the auto-generated functor's `operator()` method**, which is the structured kernel wrapper code generated from `torchgen/dest/register_dispatch_key.py` template.

## Compilation Fixes Applied

### Issue: Namespace Conflicts
The original implementation used `std::cerr` and `std::endl`, which caused compilation errors when generating code for operators in the `at::_ops::std` namespace (e.g., `_standard_gamma_grad_out`).

### Solution
Changed all prints to use **fully qualified names**:
- `std::cerr` → `::std::cerr`
- `std::endl` → `::std::endl`
- Also changed to only instrument the exact `mm` operator (`func_name == "mm"`) instead of all operators containing "mm" in their name
- Used `auto&&` instead of `auto` for result to avoid unnecessary copies and handle reference return types correctly

## To See the New Prints

You need to:

1. **Regenerate** the C++ code from templates:
   ```bash
   python -m torchgen.gen \
     --source-path aten/src/ATen \
     --install_dir aten/src/ATen
   ```

2. **Rebuild** PyTorch with the newly generated code

3. **Run** your program again

## Expected Output After Regeneration

Once regenerated and rebuilt, you should see:

```
[DEBUG at::_ops::mm::call] ===== ENTERED =====
[DEBUG at::_ops::mm::call] About to get typed handle and call dispatcher
[DEBUG at::_ops::mm::call] About to call op.call()
[DEBUG Dispatcher::call] op=aten::mm
[DEBUG OperatorEntry::lookup] op=aten::mm, keyset=DispatchKeySet(Meta, AutogradMeta)
[DEBUG KernelFunction::callBoxed] dispatchKeySet=DispatchKeySet(Meta)
[DEBUG BoxedKernel::callBoxed] Function pointer address: 0x...
[BoxedKernel LAMBDA] >>>>> Calling functor operator() <<<<<
[DEBUG structured_mm wrapper] ===== ENTERED WRAPPER =====
[DEBUG structured_mm wrapper] Function: mm
[DEBUG structured_mm wrapper] Creating op instance...
[DEBUG structured_mm] About to call op.meta()
[DEBUG TORCH_META_FUNC(mm)] ========== ENTERED MM META FUNCTION ==========
... (rest of TORCH_META_FUNC prints)
[DEBUG TORCH_META_FUNC(mm)] Meta function complete - output shape set
[DEBUG structured_mm] op.meta() RETURNED
[DEBUG structured_mm wrapper] ===== EXITING WRAPPER =====
[BoxedKernel LAMBDA] >>>>> Functor operator() RETURNED <<<<<
[DEBUG at::_ops::mm::call] op.call() RETURNED
[DEBUG at::_ops::mm::call] ===== EXITING =====
```

The hang will now be pinpointed to a specific line between two consecutive debug prints!

