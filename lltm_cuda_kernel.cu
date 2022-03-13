#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

    template <typename scalar_t>
    __global__ void lltm_cuda_forward_kernel(
        const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> input,
        const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> output) {
        //batch index
        // column index
        const int c = blockIdx.x * blockDim.x + threadIdx.x;

    }

} // namespace

std::vector<torch::Tensor> lltm_cuda_forward(
    torch::Tensor input,
    torch::Tensor output) {
 
    const int threads = 1024;
    const dim3 blocks(10);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "lltm_forward_cuda", ([&] {
        lltm_cuda_forward_kernel<scalar_t> << <blocks, threads >> > (
            input.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            output.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>());
        }));

    return { input,output };
}

