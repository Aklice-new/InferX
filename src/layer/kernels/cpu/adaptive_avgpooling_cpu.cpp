#include "core/common.h"
#include "layer/kernels/adaptive_avgpooling.h"

namespace inferx
{

namespace layer
{

StatusCode AdaptiveAvgPoolingLayer::forward_cpu()
{
    auto input = inputs_[0];
    auto output = outputs_[0];
    auto shapes = input->shapes();
    const uint32_t batch = shapes[0];
    const uint32_t channel = shapes[1];
    const uint32_t h = shapes[2];
    const uint32_t w = shapes[3];

    const uint32_t stride_h = h / output_height_;
    const uint32_t stride_w = w / output_width_;

#pragma omp parallel for num_threads(batch)
    for (int b = 0; b < batch; b++)
    {
        // from pytroch implementation
        // https://github.com/pytorch/pytorch/blob/51861cc9b19d9c483598e39932661822a826d3a2/aten/src/ATen/native/AdaptiveAveragePooling.cpp#L12
        for (int c = 0; c < channel; c++)
        {
            auto input_data_ptr = input->ptr<float>() + b * channel * h * w + c * h * w;
            auto output_data_ptr = output->ptr<float>() + b * channel * output_height_ * output_width_
                + c * output_height_ * output_width_;
            for (int oh = 0; oh < output_height_; oh++)
            {
                int istart_h = oh * h / output_height_;
                int iend_h = ((oh + 1) * h + output_height_ - 1) / output_height_;
                int k_h = iend_h - istart_h;
                for (int ow = 0; ow < output_width_; ow++)
                {
                    int istart_w = ow * w / output_width_;
                    int iend_w = ((ow + 1) * w + output_width_ - 1) / output_width_;
                    int k_w = iend_w - istart_w;
                    float sum = 0;
                    for (int ih = istart_h; ih < iend_h; ih++)
                    {
                        for (int iw = istart_w; iw < iend_w; iw++)
                        {
                            sum += input_data_ptr[ih * w + iw];
                        }
                    }
                    output_data_ptr[oh * output_width_ + ow] = sum / (k_h * k_w);
                }
            }
        }
    }

    return StatusCode::Success;
}
} // namespace layer
} // namespace inferx