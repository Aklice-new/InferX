#include "layer/kernels/relu.h"
#include "layer/layer_factory.h"

namespace inferx
{
namespace layer
{

ReluLayer::ReluLayer(std::string layer_name)
    : ActivationLayer(layer_name)
{
    activation_type_ = ActivationType::ActivationType_Relu;
}

Layer* createReluInstance(std::string layer_name)
{
    return new ReluLayer(layer_name);
}

LayerRegisterWrapper ReluLayer_Register(createReluInstance, "nn.Relu");

} // namespace layer
} // namespace inferx