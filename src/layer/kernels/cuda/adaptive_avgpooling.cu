#include "core/common.h"
#include "layer/kernels/adaptive_avgpooling.h"

namespace inferx
{
namespace layer
{
using namespace inferx::core;
/*
cuda kernel的实现方式和cpu的类似，每个thread负责output中一个元素的计算
*/

#define START_IND(a, b, c) ((int64_t)((a / b) * c + ((a % b) * c) / b))
#define END_IND(a, b, c) (1 + ((int64_t)(a + 1) * c - 1) / b)

template <typename DType>
__global__ void adaptive_avg_pooling_kernel(DType* input_data, DType* output_data, uint32_t iH, uint32_t iW,
    uint32_t oH, uint32_t oW, uint32_t iStride, uint32_t oStride)
{
    // grid
    int iplane = blockIdx.x;
    int oplane = iplane;
    input_data = input_data + iplane * iStride;
    output_data = output_data + oplane * oStride;

    int ostartH = blockDim.y * blockIdx.y + threadIdx.y;
    int oendH = oH;
    const int ostepH = blockDim.y * gridDim.y;

    int ostartW = threadIdx.x;
    int oendW = oW;
    const int ostepW = blockDim.x;

    // 循环处理当前线程需要处理的那些个元素
    for (int oh = ostartH; oh < oendH; oh += ostepH)
    {
        int istartH = START_IND(oh, oH, iH);
        int iendH = END_IND(oh, oH, iH);
        int kH = iendH - istartH;
        for (int ow = ostartW; ow < oendW; ow += ostepW)
        {
            int istartW = START_IND(ow, oW, iW);
            int iendW = END_IND(ow, oW, iW);
            int kW = iendW - istartW;
            DType sum = 0;
            for (int ih = istartH; ih < iendH; ih++)
            {
                for (int iw = istartW; iw < iendW; iw++)
                {
                    sum += input_data[ih * iW + iw];
                }
            }
            output_data[oh * oW + ow] = sum / kH / kW;
        }
    }

    return;
}

StatusCode CUDAAdaptiveAvgPooling_ForwardImp(const Tensor::TensorPtr input, Tensor::TensorPtr output)
{

    DataType dtype = input->dtype();
    auto in_shape = input->shapes();
    auto out_shape = output->shapes();
    uint32_t iN = in_shape[0];
    uint32_t iC = in_shape[1];
    uint32_t iH = in_shape[2];
    uint32_t iW = in_shape[3];
    uint32_t oH = out_shape[2];
    uint32_t oW = out_shape[3];
    uint32_t iStride = iH * iW;
    uint32_t oStride = oH * oW;
    /*
    关于线程和对应处理的数据的划分参考的是pytorch中的实现，大致的逻辑如下：
    数据格式为： NCHW
    grid 2D :  x : N * C  y : max(16 / H, 1)
    block 2D : x : 32  y : 8
    然后每个线程负责计算output中元素的值
    */
    dim3 grid(iN * iC, std::max(16 / (int) oH, 1));
    dim3 block(32, 8);
    switch (dtype)
    {
    case DataType::DataTypeFloat32:
        adaptive_avg_pooling_kernel<float>
            <<<grid, block>>>(input->ptr<float>(), output->ptr<float>(), iH, iW, oH, oW, iStride, oStride);
        break;
    case DataType::DataTypeInt32:
        adaptive_avg_pooling_kernel<int>
            <<<grid, block>>>(input->ptr<int>(), output->ptr<int>(), iH, iW, oH, oW, iStride, oStride);
        break;
    default: return StatusCode::NotImplemented;
    }

    return StatusCode::NotImplemented;
}
// {
//     uint32_t iN = in_shape[0];
//     uint32_t iC = in_shape[1];
//     uint32_t iH = in_shape[2];
//     uint32_t iW = in_shape[3];
//     uint32_t oH = out_shape[2];
//     uint32_t oW = out_shape[3];
//     uint32_t iStride = iH * iW;
//     uint32_t oStride = oH * oW;
//     /*
//     NCHW
//     N : gridDim.x
//     C : gridDim.y
//     H : blockDim.x
//     W : blockDim.y
//     */
//     dim3 grid(iN, iC);
//     dim3 block(oH, oW);
//     switch (dtype)
//     {
//     case inferx::core::DataType::DataTypeFloat32:
//         adaptive_avg_pooling_kernel<float><<<grid, block>>>(
//             reinterpret_cast<float*>(input), reinterpret_cast<float*>(output), iH, iW, oH, oW, iStride, oStride);
//         break;
//     case inferx::core::DataType::DataTypeInt32: adaptive_avg_pooling_kernel<int><<<grid, block>>>(); break;
//     default: return StatusCode::NotImplemented;
//     }

//     return StatusCode::NotImplemented;
// }
StatusCode AdaptiveAvgPoolingLayer::forward_gpu()
{
    DataType dtype = inputs_[0]->dtype();
    auto in_shape = inputs_[0]->shapes();
    auto out_shape = outputs_[0]->shapes();
    CUDAAdaptiveAvgPooling_ForwardImp(inputs_[0], outputs_[0]);

    return StatusCode::Success;
}
} // namespace layer
} // namespace inferx