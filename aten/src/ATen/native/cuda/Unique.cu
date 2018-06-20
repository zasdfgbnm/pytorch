#include "ATen/ATen.h"

#include <tuple>
#include <thrust/unique.h>
#include <thrust/sort.h>
#include <iostream>

namespace at {
namespace native{

namespace {

template <typename scalar_t>
__global__ void inverse_indices_kernel(const scalar_t* input_data,
    int64_t n,
    const scalar_t* output_data,
    const scalar_t* output_end,
    int64_t* inverse_indices_data) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;
  for (int64_t i = idx; i < n; i += stride) {
    for (const scalar_t* p = output_data; p < output_end, p++) {
      if (*p == input_data[i]) {
        inverse_indices_data[i] = p - output_data;
        break;
      }
    }
  }
}

template <typename scalar_t>
std::tuple<Tensor, Tensor> _unique_cuda_template(
    const Tensor& self,
    const bool return_inverse) {
  cudaStream_t stream = globalContext().getCurrentCUDAStream();
  auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
  auto policy = thrust::cuda::par(allocator).on(stream);

  const Tensor& input = self.contiguous();
  int64_t n = input.numel();
  const scalar_t* input_data = input.data<scalar_t>();
  Tensor output = self.copy();
  scalar_t* output_data = output.data<scalar_t>();
  thrust::sort(policy, output_data, output_data + n);
  scalar_t* output_end = thrust::unique(policy, output_data, output_data + n);
  output.resize_(output_end - output_data);

  Tensor inverse_indices = at::empty({0}, self.type().toScalarType(kLong));
  if (return_inverse) {
    inverse_indices.resize_(input.sizes());
    int64_t* inverse_indices_data = inverse_indices.data<int64_t>();
    inverse_indices_kernel<<<(n + 512 - 1) / 512, 512, 0, stream>>>(
      input_data, n, output_data, output_end, inverse_indices_data);
  }
  return std::make_tuple(input, inverse_indices);
}
} // namespace

std::tuple<Tensor, Tensor>
_unique_cuda(const Tensor& self, const bool sorted, const bool return_inverse) {
  return AT_DISPATCH_ALL_TYPES(self.type(), "unique", [&] {
    // The current CUDA implementation of unique always sort due to the
    // lack of hashtable implementation in thrust
    return _unique_cuda_template<scalar_t>(self, return_inverse);
  });
}

}  // namespace native
}  // namespace at
