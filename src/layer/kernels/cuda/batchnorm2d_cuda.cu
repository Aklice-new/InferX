#include "core/common.h"
#include "core/cuda_helper.h"
#include "core/tensor.h"
#include "layer/kernels/batchnorm2d.h"
#include <__clang_cuda_builtin_vars.h>
#include <cmath>
#include <glog/logging.h>
#include <memory>
#include <sys/types.h>
namespace inferx
{
namespace layer
{
using namespace inferx::core;

__global__ void batchnormal2d_kernel(uint32_t batch, uint32_t channel, uint32_t h, uint32_t w, const float* intput_ptr,
    float* output_ptr, const float* mean_ptr, const float* var_ptr, const float* affine_gamma_ptr,
    const float* affine_beta_ptr)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= batch * channel * h * w)
        return;
    int b = idx / (channel * h * w);
    int c = idx / (h * w);
    int row = idx / w;
    int col = idx % w;
    output_ptr[idx]
        = (intput_ptr[idx] - mean_ptr[c]) / sqrt(var_ptr[c] + 1e-6) * affine_gamma_ptr[c] + affine_beta_ptr[c];
}

StatusCode BatchNorm2DLayer::forward_gpu()
{
    auto input = inputs_[0];
    auto output = outputs_[0];
    auto shapes = input->shapes();
    const uint32_t batch = shapes[0];
    const uint32_t channel = shapes[1];
    const uint32_t h = shapes[2];
    const uint32_t w = shapes[3];
    CHECK_EQ(num_features_, channel) << "BatchNorm2DLayer num_features_ not equal to channel size.";
    CHECK_EQ(num_features_, affine_gamma_.size()) << "BatchNorm2DLayer num_features_ not equal to weights size.";
    CHECK_EQ(num_features_, affine_beta_.size()) << "BatchNorm2DLayer num_features_ not equal to bias size.";

    dim3 threads_per_block = 256;
    dim3 block_per_grid = batch * channel * h * w + 256 - 1 / 256;

    // every thread for an element

    auto gamma_beta_shape = {num_features_};

    Tensor::TensorPtr affine_beta = std::make_shared<Tensor>(DataTypeFloat32, gamma_beta_shape);
    Tensor::TensorPtr affine_gamma = std::make_shared<Tensor>(DataTypeFloat32, gamma_beta_shape);
    affine_beta->copy_from(affine_beta_.data(), affine_beta_.size());
    affine_gamma->copy_from(affine_gamma_.data(), affine_beta_.size());

    // transfer data to gpu
    input->to_cuda();
    output->to_cuda();
    mean_->to_cuda();
    var_->to_cuda();
    affine_gamma->to_cuda();
    affine_beta->to_cuda();

    batchnormal2d_kernel<<<block_per_grid, threads_per_block>>>(batch, channel, h, w, input->ptr<float>(),
        output->ptr<float>(), mean_->ptr<float>(), var_->ptr<float>(), affine_gamma->ptr<float>(),
        affine_beta->ptr<float>());
    cudaCheck(cudaGetLastError());
    return StatusCode::Success;
}
} // namespace layer
} // namespace inferx