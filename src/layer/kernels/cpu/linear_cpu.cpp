/**
 * @file linear_cpu.cpp
 * @author Aklice (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2024-10-10
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "core/common.h"
#include "layer/kernels/linear.h"

#include <cstdint>
#include <glog/logging.h>

namespace inferx
{
namespace layer
{

/*
 * @brief general matrix vector multiplication
 *
 * @tparam Dtype
 * @param input_vec  [1, M]
 * @param input_matrix   [K, M]
 * @param M
 * @param K
 */
template <typename Dtype>
void gemv_with_bias(
    Dtype* input_vec, Dtype* input_matrix, Dtype* input_bias, Dtype* output, uint32_t M, uint32_t K, bool with_bias)
{

#pragma omp parallel for num_threads(K)
    for (uint32_t i = 0; i < K; i++)
    {
        output[i] = 0;
        for (uint32_t j = 0; j < M; j++)
        {
            output[i] += input_vec[j] * input_matrix[i * M + j];
        }
        if (with_bias)
        {
            output[i] += input_bias[i];
        }
    }
}

StatusCode LinearLayer::forward_cpu()
{
    auto input = this->inputs_[0];
    auto output = this->outputs_[0];
    const auto input_shape = input->shapes();
    CHECK(input_shape.size() == 2) << "Linear operator input shape must be 2.";
    CHECK(input_shape[0] == 1) << "Linear operator input shape must be [1, M].";
    const auto M = input_shape[1];
    const auto weight_shape = this->weight_->shapes();
    CHECK(weight_shape.size() == 2) << "Linear operator weight shape must be 2.";
    CHECK(weight_shape[1] == M) << "Linear operator weight shape must be [K, M].";
    const auto K = weight_shape[1];

    gemv_with_bias<float>(input->ptr<float>(), this->weight_->ptr<float>(), this->bias_->ptr<float>(),
        output->ptr<float>(), M, K, use_bias_);

    return StatusCode::Success;
}

} // namespace layer
} // namespace inferx