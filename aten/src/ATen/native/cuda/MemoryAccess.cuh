#pragma once

#include <cstdint>
#include <type_traits>
#include <c10/util/Exception.h>
#include <c10/macros/Macros.h>

// References:
// https://devblogs.nvidia.com/cuda-pro-tip-increase-performance-with-vectorized-memory-access/

namespace at { namespace native { namespace memory {

// What does the `static_unroll` do?
//
// We want to do something like:
//
//    using args_t = typename traits::ArgsTuple;
//    args_t args;
//    #pragma unroll
//    for (int i = 0; i < traits::arity; i++) {
//      std::get<i>(args) = ....
//    }
//
// but unfortunately the above code does not work because
// the template argument has to be a compile time constant
// so `static_unroll` is created to simulate `#pragma unroll`
// using template metaprogramming.

template<template<int i> typename func, int end, int current=0>
struct static_unroll {
  template<typename... Args>
  static inline C10_HOST_DEVICE void with_args(Args&&... args) {
    func<current>::apply(std::forward<Args>(args)...);
    static_unroll<func, end, current+1>::with_args(args...);
  }
};

template<template<int i> typename func, int end>
struct static_unroll<func, end, end> {
  template<typename... Args>
  static inline C10_HOST_DEVICE void with_args(Args... args) {}
};

// aligned vector generates vectorized load/store on CUDA
template<typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vector {
  scalar_t val[vec_size];
};

namespace policies {

struct checked_unroll {

  int remaining;

  __device__ checked_unroll(int remaining): remaining(remaining) {}

  template<typename accessor_t, typename scalar_t>
  __device__ inline void load(accessor_t to, scalar_t *from) {
    int thread_idx = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < thread_work_size; i++) {
      if (thread_idx >= remaining) {
        return;
      }
      to(i) = from[thread_idx];
      thread_idx += num_threads;
    }
  }

  template<typename scalar_t>
  __device__ inline void store(scalar_t from[], scalar_t *base, int idx) {
    int thread_idx = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < thread_work_size; i++) {
      if (thread_idx >= remaining) {
        return;
      }
      to[base + block_base_idx + thread_idx] = from[i];
      thread_idx += num_threads;
    }
  }
};

// Functions here does not do boundary check. It assumes the whole block
// has its job to do. So the reminders should be handled by the the caller
// manually.

template <int vec_size>  // vec_size: number of scalars, can be 1, 2, or 4.
struct vectorized {

  static_assert(thread_work_size % vec_size == 0, "The workload per thread must be a multiple of vec_size");
  static constexpr int loop_size = thread_work_size / vec_size;

  template<typename accessor_t, typename scalar_t>
  __device__ inline void load(accessor_t to, scalar_t *from) {
    using vec_t = aligned_vector<scalar_t, vec_size>;
    vec_t *from_ = reinterpret_cast<vec_t *>(from);
    int thread_idx = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < loop_size; i++) {
      int index = thread_idx + i * num_threads;
      vec_t v = from_[index];
      #pragma unroll
      for (int j = 0; j < vec_size; j++) {
        to(vec_size * i + j) = v.val[j];
      }
    }
  }

  template<typename scalar_t>
  __device__ inline void store(scalar_t from[], scalar_t *base, int block_base_idx) {
    using vec_t = aligned_vector<scalar_t, vec_size>;
    vec_t *to_ = reinterpret_cast<vec_t *>(base + block_base_idx);
    int thread_idx = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < loop_size; i++) {
      int index = thread_idx + i * num_threads;
      vec_t v;
      for (int j = 0; j < vec_size; j++) {
        v.val[j] = from[vec_size * i + j];
      }
      to_[index] = v;
    }
  }
};

template<int i>
struct load_with_policy_helper {
  template <typename args_t, typename policy_t, typename array_t>
  static __device__ void apply(args_t args[], policy_t policy, array_t args_data, int idx) {
    using arg_t = std::tuple_element_t<i, args_t>;
    auto args_accessor = [&args] __device__ (int index) -> arg_t & { return std::get<i>(args[index]); };
    arg_t *base = reinterpret_cast<std::tuple_element_t<i, arg_ptrs>>(args_data[i]) + idx;
    policy.load(args_accessor, std::get<i>(args_base));
  }
};

template <typename args_t, typename policy_t, typename array_t>
void load_with_policy(args_t args[], policy_t policy, array_t args_data, int idx) {
  detail::static_unroll<load_with_policy, arity>::with_args(args, policy, idx);
}

}  // namespace policies

template<int i>
struct can_vectorize_up_to_helper {
  // This is only used in host, but we will wrap this into some templates
  // which is C10_HOST_DEVICE, so we have to make this C10_HOST_DEVICE
  // in order to compile
  template <typename array_t, typename traits>
  static C10_HOST_DEVICE void apply(int &result, array_t pointers, traits _) {
    using arg_t = typename traits::template arg<i>::type;
    uint64_t address = reinterpret_cast<uint64_t>(pointers[i]);
    constexpr int vec2_alignment = std::alignment_of<aligned_vector<arg_t, 2>>::value;
    constexpr int vec4_alignment = std::alignment_of<aligned_vector<arg_t, 4>>::value;
    if (address % vec4_alignment == 0) {
      result = std::min(result, 4);
    } else if (address % vec2_alignment == 0) {
      result = std::min(result, 2);
    } else {
      result = std::min(result, 1);
    }
  }
};

template<typename func_t, typename array_t>
inline int can_vectorize_up_to(char *result_data, array_t args_data) {
  using traits = function_traits<func_t>;
  using return_t = typename traits::result_type;
  constexpr int arity = traits::arity;
  int result = memory::can_vectorize_up_to<return_t>(result_data);
  // We need to get the type for each argument of `func_t`, this can only
  // be done at compile time.
  static_unroll<can_vectorize_up_to_helper, arity>::with_args(result, args_data, traits());
  return result;
}

}}} // namespace at::native::memory
