#ifndef _UNARY_H_
#define _UNARY_H_

#include "core/common.h"

namespace inferx
{
namespace layer
{
using namespace core;

StatusCode CUDAUnary_Abs_ForwardImp(const void* input, void* output, size_t size, DataType dtype);
StatusCode CUDAUnary_Relu_ForwardImp(const void* input, void* output, size_t size, DataType dtype);
StatusCode CUDAUnary_Relu6_ForwardImp(const void* input, void* output, size_t size, DataType dtype);
StatusCode CUDAUnary_Sigmoid_ForwardImp(const void* input, void* output, size_t size, DataType dtype);
StatusCode CUDAUnary_Silu_ForwardImp(const void* input, void* output, size_t size, DataType dtype);
StatusCode CUDAUnary_HardSigmoid_ForwardImp(const void* input, void* output, size_t size, DataType dtype);
StatusCode CUDAUnary_HardSwish_ForwardImp(const void* input, void* output, size_t size, DataType dtype);
StatusCode CUDAUnary_Sqrt_ForwardImp(const void* input, void* output, size_t size, DataType dtype);
StatusCode CUDAUnary_Square_ForwardImp(const void* input, void* output, size_t size, DataType dtype);
StatusCode CUDAUnary_TanH_ForwardImp(const void* input, void* output, size_t size, DataType dtype);
StatusCode CUDAUnary_Floor_ForwardImp(const void* input, void* output, size_t size, DataType dtype);
StatusCode CUDAUnary_Ceil_ForwardImp(const void* input, void* output, size_t size, DataType dtype);
StatusCode CUDAUnary_OpNum_ForwardImp(const void* input, void* output, size_t size, DataType dtype);
StatusCode CUDAUnary_Erf_ForwardImp(const void* input, void* output, size_t size, DataType dtype);
StatusCode CUDAUnary_Sin_ForwardImp(const void* input, void* output, size_t size, DataType dtype);
StatusCode CUDAUnary_Cos_ForwardImp(const void* input, void* output, size_t size, DataType dtype);
StatusCode CUDAUnary_Round_ForwardImp(const void* input, void* output, size_t size, DataType dtype);
} // namespace layer
} // namespace inferx

#endif