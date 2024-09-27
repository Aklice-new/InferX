/**
 * @file batchnorm2d_cpu.cpp
 * @author Aklice (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2024-09-27
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "core/common.h"
#include "layer/kernels/batchnorm2d.h"
#include <glog/logging.h>

namespace inferx
{
namespace layer
{

StatusCode BatchNorm2DLayer::forward_cpu()
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

#pragma omp parallel for num_threads(batch)
    for (uint32_t b = 0; b < batch; b++)
    {
        const auto input_data_ptr = input->ptr<float>() + b * channel * h * w;
        auto output_data_ptr = output->ptr<float>() + b * channel * h * w;
        for (uint32_t c = 0; c < channel; c++)
        {
            const auto input_channel_ptr = input_data_ptr + c * h * w;
            auto output_channel_ptr = output_data_ptr + c * h * w;
            const auto mean_ptr = mean_->ptr<float>() + c;
            const auto var_ptr = var_->ptr<float>() + c;
            const auto gamma = affine_gamma_[c];
            const auto beta = affine_beta_[c];
            for (uint32_t i = 0; i < h * w; i++)
            {
                output_channel_ptr[i]
                    = (input_channel_ptr[i] - mean_ptr[0]) / std::sqrt(var_ptr[0] + eps_) * gamma + beta;
            }
        }
    }
    return StatusCode::Success;
}

} // namespace layer
} // namespace inferx