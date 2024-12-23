#include "core/common.h"
#include "layer/kernels/maxpooling.h"
#include <cstdint>
#include <glog/logging.h>
#include <limits>

namespace inferx
{

namespace layer
{

StatusCode MaxPoolingLayer::forward_cpu()
{
    auto input = inputs_[0];
    auto output = outputs_[0];
    auto shapes = input->shapes();
    auto oshapes = output->shapes();
    const uint32_t batch = shapes[0];
    const uint32_t channel = shapes[1];
    const uint32_t h = shapes[2];
    const uint32_t w = shapes[3];
    const uint32_t kernel_h_ = pooling_size_h_;
    const uint32_t kernel_w_ = pooling_size_w_;
    const uint32_t output_height_ = oshapes[2];
    const uint32_t output_width_ = oshapes[3];
    const uint32_t input_padded_h = h + 2 * padding_h_;
    const uint32_t input_padded_w = w + 2 * padding_w_;

#pragma omp parallel for num_threads(batch)
    for (int b = 0; b < batch; b++)
    {
        for (int c = 0; c < channel; c++)
        {
            auto input_data_ptr = input->ptr<float>() + b * channel * h * w + c * h * w;
            auto output_data_ptr = output->ptr<float>() + b * channel * output_height_ * output_width_
                + c * output_height_ * output_width_;
            for (int r = 0; r < input_padded_h - kernel_h_ + 1; r += stride_h_)
            {
                uint32_t output_r = r / stride_h_;
                for (int c = 0; c < input_padded_w - kernel_w_ + 1; c += stride_w_)
                {
                    uint32_t output_c = c / stride_w_;
                    float max_value = std::numeric_limits<float>::lowest();
                    for (int i = 0; i < kernel_h_; i++)
                    {
                        for (int j = 0; j < kernel_w_; j++)
                        {
                            float current_value = std::numeric_limits<float>::lowest();
                            if (i + r >= padding_h_ && j + c >= padding_w_ && i + r < h + padding_h_
                                && j + c < w + padding_w_)
                            {
                                current_value = input_data_ptr[(i + r - padding_h_) * w + j + c - padding_w_];
                            }
                            else
                            {
                                current_value = std::numeric_limits<float>::lowest();
                            }
                            max_value = std::max(max_value, current_value);
                        }
                    }
                    output_data_ptr[output_r * output_width_ + output_c] = max_value;
                }
            }
        }
    }
    return StatusCode::Success;
}

} // namespace layer
} // namespace inferx