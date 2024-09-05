#include "core/common.h"
#include <cstddef>

namespace inferx
{
namespace layer
{

using namespace core;

enum UnaryOpType
{
    Unary_Unknown = 0,
    Unary_Abs,
    Unary_Relu,
    Unary_Sigmoid,
    Unary_Relu6,
    Unary_HardSigmoid,
    Unary_HardSwish,
    Unary_Silu,
    Unary_Sqrt,
    Unary_Square,
    Unary_TanH,
    Unary_Floor,
    Unary_Ceil,
    Unary_OpNum,
    Unary_Erf,
    Unary_Sin,
    Unary_Cos,
    Unary_Round
};
template <UnaryOpType OpT, typename Dtype>
__device__ __inline__ Dtype unary(const Dtype& in_val);

/**
 * @brief 这里只提供了float类型的实现，如果需要其他类型的实现，需要自行添加
 *
 */

template <>
__device__ __inline__ float unary<Unary_Abs, float>(const float& in_val)
{
    return fabs(in_val);
}

template <>
__device__ __inline__ float unary<Unary_Relu, float>(const float& in_val)
{
    return in_val > 0 ? in_val : 0;
}

template <>
__device__ __inline__ float unary<Unary_Relu6, float>(const float& in_val)
{
    return in_val > 0 ? (in_val < 6 ? in_val : 6) : 0;
}

template <>
__device__ __inline__ float unary<Unary_Sigmoid, float>(const float& in_val)
{
    return 1 / (1 + exp(-in_val));
}

template <>
__device__ __inline__ float unary<Unary_HardSigmoid, float>(const float& in_val)
{
    return in_val > 3 ? 1 : (in_val < -3 ? 0 : 0.2 * in_val + 0.5);
}

template <>
__device__ __inline__ float unary<Unary_HardSwish, float>(const float& in_val)
{
    return in_val * (in_val > 3 ? 1 : (in_val < -3 ? 0 : 0.2 * in_val + 0.5));
}

template <>
__device__ __inline__ float unary<Unary_Silu, float>(const float& in_val)
{
    return in_val / (1 + exp(-in_val));
}

template <>
__device__ __inline__ float unary<Unary_Sqrt, float>(const float& in_val)
{
    return sqrt(in_val);
}

template <>
__device__ __inline__ float unary<Unary_Square, float>(const float& in_val)
{
    return in_val * in_val;
}

template <>
__device__ __inline__ float unary<Unary_TanH, float>(const float& in_val)
{
    return tanh(in_val);
}

template <>
__device__ __inline__ float unary<Unary_Floor, float>(const float& in_val)
{
    return floor(in_val);
}

template <>
__device__ __inline__ float unary<Unary_Ceil, float>(const float& in_val)
{
    return ceil(in_val);
}

template <>
__device__ __inline__ float unary<Unary_Erf, float>(const float& in_val)
{
    return erf(in_val);
}

template <>
__device__ __inline__ float unary<Unary_Sin, float>(const float& in_val)
{
    return sin(in_val);
}

template <>
__device__ __inline__ float unary<Unary_Cos, float>(const float& in_val)
{
    return cos(in_val);
}

template <>
__device__ __inline__ float unary<Unary_Round, float>(const float& in_val)
{
    return round(in_val);
}

/**
 * @brief 这里只提供了float类型的实现，如果需要其他类型的实现，需要自行添加
 *
 */
template <UnaryOpType OpT, typename Dtype>
__global__ void unary_kernel(const Dtype* input, Dtype* output, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output[idx] = unary<OpT, Dtype>(input[idx]);
    }
}

#define UNARY_CUDA_KERNEL_INSTALL(TYPE)                                                                                \
    StatusCode CUDAUnary_##TYPE##_ForwardImp(                                                                          \
        const void* input, void* output, size_t size, inferx::core::DataType dtype)                                    \
    {                                                                                                                  \
        dim3 block_size = 256;                                                                                         \
        dim3 grid_size = (size + block_size.x - 1) / block_size.x;                                                     \
        switch (dtype)                                                                                                 \
        {                                                                                                              \
        case inferx::core::DataType::DataTypeFloat32:                                                                  \
            unary_kernel<Unary_##TYPE, float><<<grid_size, block_size>>>((const float*) input, (float*) output, size); \
            break;                                                                                                     \
        default: return inferx::core::StatusCode::NotImplemented;                                                      \
        }                                                                                                              \
        return inferx::core::StatusCode::Success;                                                                      \
    }

UNARY_CUDA_KERNEL_INSTALL(Abs)
UNARY_CUDA_KERNEL_INSTALL(Relu)
UNARY_CUDA_KERNEL_INSTALL(Sigmoid)
UNARY_CUDA_KERNEL_INSTALL(Relu6)
UNARY_CUDA_KERNEL_INSTALL(HardSigmoid)
UNARY_CUDA_KERNEL_INSTALL(HardSwish)
UNARY_CUDA_KERNEL_INSTALL(Silu)
UNARY_CUDA_KERNEL_INSTALL(Sqrt)
UNARY_CUDA_KERNEL_INSTALL(Square)
UNARY_CUDA_KERNEL_INSTALL(TanH)
UNARY_CUDA_KERNEL_INSTALL(Floor)
UNARY_CUDA_KERNEL_INSTALL(Ceil)
UNARY_CUDA_KERNEL_INSTALL(Erf)
UNARY_CUDA_KERNEL_INSTALL(Sin)
UNARY_CUDA_KERNEL_INSTALL(Cos)
UNARY_CUDA_KERNEL_INSTALL(Round)
} // namespace layer
} // namespace inferx