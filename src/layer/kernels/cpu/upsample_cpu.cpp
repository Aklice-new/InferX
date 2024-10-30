/**
 * @file upsample_cpu.cpp
 * @author Aklice (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2024-10-24
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "core/common.h"
#include "layer/kernels/upsample.h"

#include <cstdint>
#include <glog/logging.h>
#include <sys/types.h>

namespace inferx
{
namespace layer
{
StatusCode UpsampleLayer::forward_cpu()
{
    auto input = inputs_[0];
    auto output = outputs_[0];
    auto in_shapes = input->shapes();
    auto out_shapes = output->shapes();
    CHECK_EQ(in_shapes.size(), 4) << "Input tensor shape must be 4-D.";
    CHECK_EQ(out_shapes.size(), 4) << "Output tensor shape must be 4-D.";
    auto N = in_shapes[0];
    auto C = in_shapes[1];
    auto H = in_shapes[2];
    auto W = in_shapes[3];

    auto input_batch_size = C * H * W;
    auto input_channels_size = H * W;
    auto output_batch_size = C * H * W * scale_factor_h_ * scale_factor_w_;
    auto output_channels_size = H * W * scale_factor_h_ * scale_factor_w_;

    CHECK_EQ(H * scale_factor_h_, out_shapes[2]) << "Batch size must be the same.";
    CHECK_EQ(W * scale_factor_w_, out_shapes[3]) << "Channel size must be the same.";

    switch (mode_)
    {
    case Mode::Nearest:
    {
#pragma omp parallel for
        for (uint32_t n = 0; n < N; n++)
        {
            for (uint32_t c = 0; c < C; c++)
            {
                for (uint32_t h = 0; h < H; h++)
                {
                    auto input_ptr = input->ptr<float>() + n * input_batch_size + c * input_channels_size + h * W;
                    auto out_row = h * scale_factor_h_;
                    for (uint32_t s_h = 0; s_h < scale_factor_h_; s_h++)
                    {
                        auto output_row_ptr = output->ptr<float>() + n * output_batch_size + c * output_channels_size
                            + out_row * W * scale_factor_w_;
                        for (uint32_t w = 0; w < W; w++)
                        {
                            auto out_col = w * scale_factor_w_;
                            auto input_value = input_ptr[w];
                            for (uint32_t s_w = 0; s_w < scale_factor_w_; s_w++)
                            {
                                output_row_ptr[out_col + s_w] = input_value;
                            }
                        }
                    }
                }
            }
        }
    }
    break;
    case Mode::Bilinear:
    {
    }
    break;
    default:
    {
        LOG(ERROR) << "Upsample mode not supported.";
        return StatusCode::Failed;
    }
    break;
    }
    return StatusCode::Success;
}
} // namespace layer
} // namespace inferx