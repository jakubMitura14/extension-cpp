#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void lltm_cuda_forward(
    torch::Tensor input,
    torch::Tensor output, int xDim, int yDim, int zDim);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void lltm_forward(
    torch::Tensor input,
    torch::Tensor output, int xDim, int yDim, int zDim) {

    CHECK_INPUT(input);
    CHECK_INPUT(output);

    printf(" \n beefore lltm_cuda_forward in cpp \n");

    lltm_cuda_forward(input, output, xDim, yDim, zDim);

    printf(" \n aaafter lltm_cuda_forward in cpp \n");

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forwardB", &lltm_forward, "LLTM forward (CUDA)");
}
