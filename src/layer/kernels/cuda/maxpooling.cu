#include "core/common.h"
#include "layer/kernels/maxpooling.h"
#include <cstdint>
#include <limits>

namespace inferx
{

namespace layer
{
using namespace core;

#define CEIL_DIV(x, y) (((x) + (y) -1) / (y))

template <typename DType>
__global__ void max_pooling_kernel(DType* input_data, DType* output_data, uint32_t iH, uint32_t iW, uint32_t oH,
    uint32_t oW, uint32_t padding_h_, uint32_t padding_w_, uint32_t kernel_h_, uint32_t kernel_w_, uint32_t stride_h_,
    uint32_t stride_w_, uint32_t input_padded_h, uint32_t input_padded_w, uint32_t iStride, uint32_t oStride)
{
    const int iplane = blockIdx.x;
    const int oplane = iplane;
    input_data = input_data + iplane * iStride;
    output_data = output_data + oplane * oStride;
    uint32_t i_start_h = threadIdx.x * stride_h_;
    uint32_t i_start_w = threadIdx.y * stride_w_;
    // uint32_t i_end_h = i_start_h + kernel_h_;
    // uint32_t i_end_w = i_start_w + kernel_w_;

    uint32_t output_row = uint32_t(i_start_h / stride_h_);

    uint32_t output_col = uint32_t(i_start_w / stride_w_);
    float max_value = std::numeric_limits<float>::lowest();
    for (uint32_t w = 0; w < kernel_w_; w++)
    {
        for (uint32_t h = 0; h < kernel_h_; h++)
        {
            float current_value = 0.f;
            if (i_start_h + h >= padding_h_ && i_start_w + w < padding_w_ && h + i_start_h < iH + padding_h_
                && i_start_w + w >= iW + padding_w_)
            {
                current_value = input_data[(i_start_h + h - padding_h_) * iW + (i_start_w + w - padding_w_)];
            }
            else
            {
                current_value = std::numeric_limits<float>::lowest();
            }
            max_value = max_value > current_value ? max_value : current_value;
        }
    }
    output_data[output_row * oW + output_col] = max_value;
} // namespace layer

StatusCode CUDAMaxPooling_ForwardImp(const Tensor::TensorPtr input, Tensor::TensorPtr output, uint32_t padding_h,
    uint32_t padding_w, uint32_t kernel_h, uint32_t kernel_w, uint32_t stride_h, uint32_t stride_w)
{
    DataType dtype = input->dtype();
    auto in_shape = input->shapes();
    auto out_shape = output->shapes();
    const uint32_t iN = in_shape[0];
    const uint32_t iC = in_shape[1];
    const uint32_t iH = in_shape[2];
    const uint32_t iW = in_shape[3];
    const uint32_t oH = out_shape[2];
    const uint32_t oW = out_shape[3];
    const uint32_t input_padded_h = iH + 2 * padding_h;
    const uint32_t input_padded_w = iW + 2 * padding_w;

    /*
    关于线程和对应处理的数据的划分参考的是pytorch中的实现，大致的逻辑如下：
    数据格式为： NCHW
    grid 2D :  x : N * C
    block 2D : x : ceil(input_padded_h - pooling_h + 1 / stride_h_)  y : ceil(input_padded_w - pooling_w + 1 /
    stride_w_)
    然后每个线程负责计算一个output中元素的值
    */
    dim3 grid(iN * iC);
    dim3 block(CEIL_DIV(input_padded_h - kernel_h + 1, stride_h), CEIL_DIV(input_padded_w - kernel_w + 1, stride_w));

    switch (dtype)
    {
    case DataType::DataTypeFloat32:
        max_pooling_kernel<float><<<grid, block>>>(input->ptr<float>(), output->ptr<float>(), iH, iW, oH, oW, padding_h,
            padding_w, kernel_h, kernel_w, stride_h, stride_w, input_padded_h, input_padded_w, iH * iW, oH * oW);
        break;
    default: return StatusCode::Failed;
    }
    return StatusCode::Success;
}
StatusCode MaxPoolingLayer::forward_gpu()
{
    return StatusCode::Success;
}
} // namespace layer
} // namespace inferx