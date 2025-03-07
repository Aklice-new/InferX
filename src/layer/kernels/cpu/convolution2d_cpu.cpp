#include "core/common.h"
#include "core/tensor.h"
#include "layer/kernels/convolution2d.h"

#include <cstdint>
#include <glog/logging.h>
#include <memory>

namespace inferx
{

namespace layer
{

template <typename Dtype>
void im2col_cpu(const Dtype* input_data, const uint32_t in_group_size, const uint32_t input_h, const uint32_t input_w,
    const uint32_t kernel_h, const uint32_t kernel_w, const uint32_t stride_h, const uint32_t stride_w,
    const uint32_t dilation_h, const uint32_t dilation_w, const uint32_t padding_h, const uint32_t padding_w,
    const uint32_t output_h, const uint32_t output_w, Dtype* output_data)
{
    const uint32_t channel_size = input_h * input_w;
    const uint32_t kernel_size = kernel_h * kernel_w;
    for (uint32_t c = 0; c < in_group_size; c++, input_data += channel_size)
    {
        // 遍历kernel中每一个元素
        for (uint32_t kernel_row = 0; kernel_row < kernel_h; kernel_row++)
        {
            for (uint32_t kernel_col = 0; kernel_col < kernel_w; kernel_col++)
            {
                // 计算起始的行
                uint32_t input_row = -padding_h + kernel_row * dilation_h;
                // 因为output中每个元素都有当前这个kernel中的元素的贡献，所以遍历output的每一个元素
                for (uint32_t output_rows = output_h; output_rows--; input_row += stride_h)
                {
                    // 这时说明找到的是矩阵外面填充的部分
                    if (input_row < 0 || input_row >= input_h)
                    {
                        for (uint32_t output_cols = output_w; output_cols--;)
                        {
                            *(output_data++) = 0;
                        }
                    }
                    else
                    {
                        // 找到了对应的列
                        uint32_t input_col = -padding_w + kernel_col * dilation_w;
                        for (uint32_t output_cols = output_w; output_cols--; input_col += stride_w)
                        {
                            if (input_col < 0 || input_col >= input_w)
                            {
                                *(output_data++) = 0;
                            }
                            else
                            {
                                // 这里就是已经找到的对应的元素
                                *(output_data++) = input_data[input_row * input_w + input_col];
                            }
                        }
                    }
                }
            }
        }
    }
}

template <typename Dtype>
void gemm_with_bais_cpu(
    const Dtype* A, const Dtype* B, const Dtype* bias, const uint32_t M, const uint32_t N, const uint32_t K, Dtype* D)
{
    // A [M, K]
    // B [K, N]
    for (uint32_t m = 0; m < M; m++)
    {
        for (uint32_t n = 0; n < N; n++)
        {
            Dtype sum = 0;
            for (uint32_t k = 0; k < K; k++)
            {
                sum += A[m * K + k] * B[k * N + n];
            }
            D[m * N + n] = sum + bias[m];
        }
    }
}

StatusCode Convolution2DLayer::forward_cpu()
{
    auto input = inputs_[0];
    auto output = outputs_[0];
    const auto weight = weights_[0];
    auto bias = bias_[0];

    auto kernel_h = kernel_h_;
    auto kernel_w = kernel_w_;
    auto stride_h = stride_h_;
    auto stride_w = stride_w_;
    auto dilation_h = dilation_h_;
    auto dilation_w = dilation_w_;
    auto padding_h = padding_h_;
    auto padding_w = padding_w_;
    auto input_shapes = input->shapes();

    CHECK(input_shapes.size() == 4) << "Input tensor shape must be 4-D. format [N, C, H, W]";
    CHECK(out_channels_ % groups_ == 0) << "out_channels must be divisible by groups.";

    auto input_h = input_shapes[2];
    auto input_w = input_shapes[3];
    uint32_t batch = input_shapes[0];
    uint32_t group_size = in_channels_ / groups_;
    // https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    this->output_h_ = (input_h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    this->output_w_ = (input_w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    std::vector<uint32_t> output_shapes = {batch, out_channels_, output_h_, output_w_};
    std::vector<uint32_t> img_col_shapes = {group_size * kernel_h_ * kernel_w_, output_h_ * output_w_};
    std::vector<uint32_t> weight_col_shapes = {out_channels_ / groups_, group_size * kernel_h_ * kernel_w_};
    Tensor::TensorPtr img_col = std::make_shared<Tensor>(DataTypeFloat32, img_col_shapes);
    // for debug
    this->img2col_ = img_col;
    // make sure the col_buffer is on the same device with input
    img_col->apply_data();

    for (uint32_t b = 0; b < batch; b++)
    {
        for (uint32_t g = 0; g < groups_; g++)
        {
            // im2col
            // input shape: N, C, H, W
            // img_col shape:  in_channels * kernel_h * kernel_w, output_h * output_w
            Tensor::TensorPtr weight_col = std::make_shared<Tensor>(DataTypeFloat32, weight_col_shapes);
            // weight_col->apply_data();
            im2col_cpu<float>(
                input->ptr<float>() + b * in_channels_ * input_w * input_h + group_size * g * input_w * input_h,
                group_size, input_h, input_w, kernel_h, kernel_w, stride_h, stride_w, dilation_h, dilation_w, padding_h,
                padding_w, output_h_, output_w_, img_col->ptr<float>());

            weight_col->copy_from(reinterpret_cast<const void*>(weight->ptr<float>()
                                      + g * (out_channels_ / groups_) * group_size * kernel_h * kernel_w),
                (out_channels_ / groups_) * group_size * kernel_h * kernel_w);
            // gemm with bias
            // weight_col shape: out_group_size, in_group_size * kernel_h * kernel_w
            // img_col shape: in_group_size * kernel_h * kernel_w, output_h * output_w
            // bias shape: out_group_size
            // output shape: out_group_size, output_h * output_w
            gemm_with_bais_cpu<float>(weight_col->ptr<float>(), img_col->ptr<float>(),
                bias->ptr<float>() + g * (out_channels_ / groups_), out_channels_ / groups_, output_h_ * output_w_,
                group_size * kernel_h * kernel_w,
                output->ptr<float>() + b * (output_h_ * output_w_ * out_channels_)
                    + g * (out_channels_ / groups_) * output_h_ * output_w_);
        }
    }
    // col2im
    // output shape: N, out_channels, output_h, output_w
    output->Reshape(output_shapes);
    return StatusCode::Success;
}

} // namespace layer
} // namespace inferx