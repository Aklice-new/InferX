#include "core/common.h"
#include "layer/kernels/activation.h"
#include "utils/utils.h"
#include <glog/logging.h>

namespace inferx
{
namespace layer
{

template <typename DType>
void relu(const DType* input, DType* output, size_t size)
{
#pragma omp parallel for num_threads(MAX_THREADS)
    for (size_t i = 0; i < size; i++)
    {
        output[i] = input[i] > 0 ? input[i] : 0;
    }
}

template <typename DType>
void relu6(const DType* input, DType* output, size_t size)
{
#pragma omp parallel for num_threads(MAX_THREADS)
    for (size_t i = 0; i < size; i++)
    {
        output[i] = input[i] > 0 ? (input[i] < 6 ? input[i] : 6) : 0;
    }
}

template <typename DType>
void sigmoid(const DType* input, DType* output, size_t size)
{
#pragma omp parallel for num_threads(MAX_THREADS)
    for (size_t i = 0; i < size; i++)
    {
        output[i] = 1 / (1 + expf(-input[i]));
    }
}

template <typename DType>
void silu(const DType* input, DType* output, size_t size)
{
#pragma omp parallel for num_threads(MAX_THREADS)
    for (size_t i = 0; i < size; i++)
    {
        output[i] = input[i] / (1 + exp(-input[i]));
    }
}

template <typename DType>
void tanH(const DType* input, DType* output, size_t size)
{
#pragma omp parallel for num_threads(MAX_THREADS)
    for (size_t i = 0; i < size; i++)
    {
        output[i] = tanh(input[i]);
    }
}

template <typename DType>
void hard_sigmoid(const DType* input, DType* output, size_t size)
{
#pragma omp parallel for num_threads(MAX_THREADS)
    for (size_t i = 0; i < size; i++)
    {
        output[i] = input[i] >= 3 ? 1 : (input[i] < -3 ? 0 : input[i] * 1.0 / 6 + 0.5);
    }
}

template <typename DType>
void hard_swish(const DType* input, DType* output, size_t size)
{
#pragma omp parallel for num_threads(MAX_THREADS)
    for (size_t i = 0; i < size; i++)
    {
        output[i] = input[i] * (input[i] > 3 ? 1 : (input[i] < -3 ? 0 : input[i] * 1.0 / 6 + 0.5));
    }
}

template <typename DType>
inline void activateFunction(const DType* input, DType* output, size_t size, ActivationType activation_type_)
{
    switch (activation_type_)
    {
    case ActivationType::ActivationType_Relu:
    {
        relu(input, output, size);
        break;
    }
    case ActivationType::ActivationType_Relu6:
    {
        relu6(input, output, size);
        break;
    }
    case ActivationType::ActivationType_Sigmoid:
    {
        sigmoid(input, output, size);
        break;
    }
    case ActivationType::ActivationType_Silu:
    {
        silu(input, output, size);
        break;
    }
    case ActivationType::ActivationType_HardSigmoid:
    {
        hard_sigmoid(input, output, size);
        break;
    }
    case ActivationType::ActivationType_HardSwish:
    {
        hard_swish(input, output, size);
        break;
    }
    case ActivationType::ActivationType_TanH:
    {
        tanH(input, output, size);
        break;
    }
    default:
    {
        break;
    }
    }
}

StatusCode ActivationLayer::forward_cpu()
{
    DataType dtype = inputs_[0]->dtype();
    size_t size = inputs_[0]->size();
    switch (dtype)
    {
    case DataType::DataTypeFloat32:
    {
        activateFunction<float>(inputs_[0]->ptr<float>(), outputs_[0]->ptr<float>(), size, activation_type_);
        break;
    }
    case DataType::DataTypeInt32:
    {
        activateFunction<int32_t>(inputs_[0]->ptr<int32_t>(), outputs_[0]->ptr<int32_t>(), size, activation_type_);
        break;
    }
    default:
    {
        return StatusCode::Failed;
    }
    }
    return StatusCode::Success;
}
} // namespace layer
} // namespace inferx
