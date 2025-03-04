/**
 * @file linear_cuda.cu
 * @author Aklice (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2024-10-10
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "core/common.h"
#include "core/cuda_helper.h"
#include "layer/kernels/linear.h"
#include <glog/logging.h>

#include "cublas_v2.h"

namespace inferx
{
namespace layer
{
using namespace inferx::core;

StatusCode LinearLayer::forward_gpu()
{
    auto input = this->inputs_[0];
    auto output = this->outputs_[0];
    const auto input_shape = input->shapes();
    CHECK(input_shape.size() == 2) << "Linear operator input shape must be 2.";
    CHECK(input_shape[0] == 1) << "Linear operator input shape must be [1, M].";
    const auto M = input_shape[1];
    const auto weight_shape = this->weights_->shapes();
    CHECK(weight_shape.size() == 2) << "Linear operator weight shape must be 2.";
    CHECK(weight_shape[1] == M) << "Linear operator weight shape must be [K, M].";
    const auto K = weight_shape[0];

    input->to_cuda();
    output->to_cuda();
    weights_->to_cuda();
    bias_->to_cuda();

    cublasHandle_t handle;
    cublasCheck(cublasCreate(&handle));

    float alpha = 1.0f;
    float beta = 0.0f;
    cublasCheck(cublasSgemv_v2(handle, CUBLAS_OP_T, M, K, &alpha, weights_->ptr<float>(), M, output->ptr<float>(), 1,
        &beta, output->ptr<float>(), 1));
    cublasCheck(cublasSaxpy_v2(handle, K, &alpha, bias_->ptr<float>(), 1, output->ptr<float>(), 1));

    return StatusCode::Success;
}
} // namespace layer
} // namespace inferx