#include "core/common.h"
#include "layer/kernels/relu.h"
#include "layer/kernels/activation_type.h"
namespace inferx
{
namespace layer
{

ReluLayer::ReluLayer(std::string layer_name)
    : Layer(layer_name)
{
    activation_type_ = ActivationType::ActivationType_Relu;
}

StatusCode ReluLayer::prepare_layer()
{
    return StatusCode::Success;
}

} // namespace layer
} // namespace inferx
