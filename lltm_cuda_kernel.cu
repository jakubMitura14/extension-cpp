#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

    template <typename scalar_t>
    __global__ void lltm_cuda_forward_kernel(
        const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> gates,
        const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> old_cell,
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> new_h,
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> new_cell,
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> input_gate,
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> output_gate,
        torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> candidate_cell) {
        //batch index
        const int n = blockIdx.y;
        // column index
        const int c = blockIdx.x * blockDim.x + threadIdx.x;

    }

} // namespace

std::vector<torch::Tensor> lltm_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor old_h,
    torch::Tensor old_cell) {
    auto X = torch::cat({ old_h, input }, /*dim=*/1);
    auto gate_weights = torch::addmm(bias, X, weights.transpose(0, 1));

    const auto batch_size = old_cell.size(0);
    const auto state_size = old_cell.size(1);

    auto gates = gate_weights.reshape({ batch_size, 3, state_size });
    auto new_h = torch::zeros_like(old_cell);
    auto new_cell = torch::zeros_like(old_cell);
    auto input_gate = torch::zeros_like(old_cell);
    auto output_gate = torch::zeros_like(old_cell);
    auto candidate_cell = torch::zeros_like(old_cell);

    const int threads = 1024;
    const dim3 blocks(10);

    AT_DISPATCH_FLOATING_TYPES(gates.type(), "lltm_forward_cuda", ([&] {
        lltm_cuda_forward_kernel<scalar_t> << <blocks, threads >> > (
            gates.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            old_cell.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            new_h.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            new_cell.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            input_gate.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            output_gate.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            candidate_cell.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>());
        }));

    return { new_h, new_cell, input_gate, output_gate, candidate_cell, X, gates };
}

