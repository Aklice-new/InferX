#include "layer/kernels/relu6.h"
#include "layer/layer_factory.h"

namespace inferx
{
namespace layer
{

Relu6Layer::Relu6Layer(std::string layer_name)
    : ActivationLayer(layer_name)
{
    activation_type_ = ActivationType::ActivationType_Relu6;
}

Layer* createRelu6Instance(std::string layer_name)
{
    return new Relu6Layer(layer_name);
}

LayerRegisterWrapper Relu6Layer_Register(createRelu6Instance, "nn.ReLU6");

} // namespace layer
} // namespace inferx