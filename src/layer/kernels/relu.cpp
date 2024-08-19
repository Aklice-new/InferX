#include "layer/kernels/relu.h"
#include "core/common.h"

namespace inferx
{
namespace layer
{

ReluLayer::ReluLayer(std::string layer_name)
    : Layer(layer_name)
{
    activation_type_ = ActivationType::ActivationType_Relu;
}

StatusCode ReluLayer::prepare_layer(
    const std::vector<Tensor::TensorPtr>& inputs, const std::vector<Tensor::TensorPtr>& outputs)
{
    return StatusCode::Success;
}

StatusCode ReluLayer::load_model(const std::map<std::string, pnnx::Attribute>& attributes)
{
    return StatusCode::Success;
}

StatusCode ReluLayer::load_param(const std::map<std::string, pnnx::Parameter>& params)
{
    return StatusCode::Success;
}
} // namespace layer
} // namespace inferx