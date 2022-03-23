#include <torch/extension.h>

#include <vector>

// CUDA  declarations

void lltm_cuda_forward(
    torch::Tensor input,
    torch::Tensor output, int xDim, int yDim, int zDim);

int getHausdorffDistance_CUDA(
    torch::Tensor goldStandard,
    torch::Tensor algoOutput
    , const  int xDim, const int yDim
    , const int zDim,const float robustnessPercent);

std::tuple<int, double>  benchmarkOlivieraCUDA(
    torch::Tensor goldStandard,
    torch::Tensor algoOutput
    , const  int xDim, const int yDim
    , const int zDim);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void lltm_forward(
    torch::Tensor input,
    torch::Tensor output
    , const int xDim, const int yDim, const int zDim) {

    CHECK_INPUT(input);
    CHECK_INPUT(output);


    lltm_cuda_forward(input, output, xDim, yDim, zDim);


}

int getHausdorffDistance(
    torch::Tensor goldStandard,
    torch::Tensor algoOutput
    , const  int xDim, const int yDim, const int zDim
    , const float robustnessPercent=1.0
) {

    CHECK_INPUT(goldStandard);
    CHECK_INPUT(algoOutput);


   return  getHausdorffDistance_CUDA(goldStandard, algoOutput, xDim, yDim, zDim, robustnessPercent);


}




std::tuple<int, double>  benchmarkOlivieraCUDAOnlyBool(
    torch::Tensor goldStandard,
    torch::Tensor algoOutput
    , const  int xDim, const int yDim
    , const int zDim) {
    return  benchmarkOlivieraCUDA(goldStandard, algoOutput, xDim, yDim, zDim);


}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forwardB", &lltm_forward, "LLTM forward (CUDA)");
    m.def("getHausdorffDistance", &getHausdorffDistance, "Basic version of Hausdorff distance");
    m.def("benchmarkOlivieraCUDA", &benchmarkOlivieraCUDA, "Algorithm by Oliviera - just for comparison sake - accept only boolean arrays  ");
}
