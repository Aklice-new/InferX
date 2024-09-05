#include "core/common.h"
#include "layer/kernels/activation.h"
#include "layer/kernels/unary.h"
namespace inferx
{
namespace layer
{

StatusCode ActivationLayer::forward_gpu()
{
    DataType dtype = inputs_[0]->dtype();
    size_t size = inputs_[0]->size();
    switch (this->activation_type_)
    {
    case ActivationType::ActivationType_Relu:
        CUDAUnary_Relu_ForwardImp(inputs_[0]->gpu_data(), outputs_[0]->gpu_data(), size, dtype);
        break;
    case ActivationType::ActivationType_Relu6:
        CUDAUnary_Relu6_ForwardImp(inputs_[0]->gpu_data(), outputs_[0]->gpu_data(), size, dtype);
        break;
    case ActivationType::ActivationType_Sigmoid:
        CUDAUnary_Sigmoid_ForwardImp(inputs_[0]->gpu_data(), outputs_[0]->gpu_data(), size, dtype);
        break;
    case ActivationType::ActivationType_TanH:
        CUDAUnary_TanH_ForwardImp(inputs_[0]->gpu_data(), outputs_[0]->gpu_data(), size, dtype);
        break;
    case ActivationType::ActivationType_Silu:
        CUDAUnary_Silu_ForwardImp(inputs_[0]->gpu_data(), outputs_[0]->gpu_data(), size, dtype);
        break;
    case ActivationType::ActivationType_HardSigmoid:
        CUDAUnary_HardSigmoid_ForwardImp(inputs_[0]->gpu_data(), outputs_[0]->gpu_data(), size, dtype);
        break;
    case ActivationType::ActivationType_HardSwish:
        CUDAUnary_HardSigmoid_ForwardImp(inputs_[0]->gpu_data(), outputs_[0]->gpu_data(), size, dtype);
        break;
    }
    return StatusCode::Success;
}

} // namespace layer
} // namespace inferx