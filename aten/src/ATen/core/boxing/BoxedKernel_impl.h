#pragma once

namespace c10 {

inline BoxedKernel::BoxedKernel() : boxed_kernel_func_(nullptr) {}

inline BoxedKernel::BoxedKernel(
    std::unique_ptr<OperatorKernel> functor,
    InternalBoxedKernelFunction* boxed_kernel_func)
    : functor_(std::move(functor)), boxed_kernel_func_(boxed_kernel_func) {}

template <BoxedKernel::BoxedKernelFunction* func>
inline void BoxedKernel::make_boxed_function(
    OperatorKernel* /*unused*/,
    const OperatorHandle& opHandle,
    DispatchKeySet /*unused*/,
    Stack* stack) {
  // Note that we're dropping the DispatchKeySet argument.
  // See Note [Plumbing Keys Through The Dispatcher 2] for details.
  func(opHandle, stack);
}

template <BoxedKernel::BoxedKernelFunction_withDispatchKeys* func>
inline void BoxedKernel::make_boxed_function(
    OperatorKernel* /*unused*/,
    const OperatorHandle& opHandle,
    DispatchKeySet ks,
    Stack* stack) {
  // See Note [Plumbing Keys Through The Dispatcher 2] for details.
  func(opHandle, ks, stack);
}

inline bool BoxedKernel::isValid() const {
  return boxed_kernel_func_ != nullptr;
}

inline bool BoxedKernel::isFallthrough() const {
  return boxed_kernel_func_ == &fallthrough_kernel;
}

inline void BoxedKernel::callBoxed(
    const OperatorHandle& opHandle,
    DispatchKeySet dispatchKeySet,
    Stack* stack) const {
  std::cerr << "\n========== BoxedKernel::callBoxed ENTRY ==========" << std::endl;
  std::cerr << "[BoxedKernel] dispatchKeySet=" << dispatchKeySet << std::endl;
  std::cerr << "[BoxedKernel] Function pointer address: " << (void*)boxed_kernel_func_ << std::endl;
  std::cerr << "[BoxedKernel] Functor pointer: " << (void*)functor_.get() << std::endl;
  std::cerr << "[BoxedKernel] OperatorHandle address: " << (void*)&opHandle << std::endl;
  std::cerr << "[BoxedKernel] Stack address: " << (void*)stack << std::endl;
  if (stack) {
    std::cerr << "[BoxedKernel] Stack size: " << stack->size() << std::endl;
    if (!stack->empty()) {
      std::cerr << "[BoxedKernel] Stack[0] type: " << stack->at(0).tagKind() << std::endl;
    }
  }
  std::cerr.flush();
  
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      boxed_kernel_func_ != nullptr,
      "Tried to call BoxedKernel::callBoxed() on an uninitialized BoxedKernel.");
  
  std::cerr << "\n[BoxedKernel] ============================================" << std::endl;
  std::cerr << "[BoxedKernel] ABOUT TO CALL THE FUNCTION POINTER" << std::endl;
  std::cerr << "[BoxedKernel] Function signature: void(*)(OperatorKernel*, const OperatorHandle&, DispatchKeySet, Stack*)" << std::endl;
  std::cerr << "[BoxedKernel] If hang occurs, it's inside this call:" << std::endl;
  std::cerr << "[BoxedKernel]   (*boxed_kernel_func_)(functor, opHandle, dispatchKeySet, stack)" << std::endl;
  std::cerr << "[BoxedKernel] ============================================\n" << std::endl;
  std::cerr.flush();
  
  // Try-catch to see if exception is thrown
  try {
    std::cerr << "[BoxedKernel] >>>>> INVOKING NOW <<<<<" << std::endl;
    std::cerr.flush();
    
    (*boxed_kernel_func_)(functor_.get(), opHandle, dispatchKeySet, stack);
    
    std::cerr << "[BoxedKernel] >>>>> RETURNED SUCCESSFULLY <<<<<" << std::endl;
    std::cerr.flush();
  } catch (const std::exception& e) {
    std::cerr << "[BoxedKernel] >>>>> EXCEPTION CAUGHT: " << e.what() << " <<<<<" << std::endl;
    std::cerr.flush();
    throw;
  } catch (...) {
    std::cerr << "[BoxedKernel] >>>>> UNKNOWN EXCEPTION CAUGHT <<<<<" << std::endl;
    std::cerr.flush();
    throw;
  }
  
  std::cerr << "[BoxedKernel] Function call completed successfully!" << std::endl;
  std::cerr << "========== BoxedKernel::callBoxed EXIT ==========\n" << std::endl;
  std::cerr.flush();
}

template <BoxedKernel::BoxedKernelFunction* func>
inline BoxedKernel BoxedKernel::makeFromFunction() {
  return BoxedKernel(
      nullptr, // no functor_ object
      &make_boxed_function<func>);
}

template <BoxedKernel::BoxedKernelFunction_withDispatchKeys* func>
inline BoxedKernel BoxedKernel::makeFromFunction() {
  return BoxedKernel(
      nullptr, // no functor_ object
      &make_boxed_function<func>);
}

inline BoxedKernel BoxedKernel::makeFallthrough() {
  return BoxedKernel(
      nullptr, // no functor_ object
      &fallthrough_kernel);
}

inline BoxedKernel BoxedKernel::makeAmbiguousAutogradOther() {
  return BoxedKernel(
      nullptr, // no functor_ object
      &ambiguous_autogradother_kernel);
}

inline BoxedKernel BoxedKernel::makeNamedNotSupported() {
  return BoxedKernel(
      nullptr, // no functor_ object
      &named_not_supported_kernel);
}

template <class KernelFunctor>
inline BoxedKernel BoxedKernel::makeFromFunctor(
    std::unique_ptr<KernelFunctor> kernelFunctor) {
  static_assert(
      std::is_base_of_v<OperatorKernel, KernelFunctor>,
      "Tried to call BoxedKernel::makeFromFunctor<KernelFunctor>, but the functor doesn't inherit from c10::OperatorKernel. Please have the functor inherit from it.");
  return BoxedKernel(
      std::move(kernelFunctor),
      [](OperatorKernel* kernel,
         const OperatorHandle& op,
         DispatchKeySet ks,
         Stack* stack) {
        (*static_cast<KernelFunctor*>(kernel))(op, ks, stack);
      });
}

inline OperatorKernel* BoxedKernel::getFunctor() const {
  return functor_.get();
}
inline BoxedKernel::InternalBoxedKernelFunction* BoxedKernel::getFnPtr() const {
  return boxed_kernel_func_;
}

} // namespace c10
