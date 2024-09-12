#include "layer/kernels/hard_swish.h"
#include "layer/layer_factory.h"

namespace inferx
{
namespace layer
{

HardSwishLayer::HardSwishLayer(std::string layer_name)
    : ActivationLayer(layer_name)
{
    activation_type_ = ActivationType::ActivationType_HardSwish;
}

Layer* createHardSwishInstance(std::string layer_name)
{
    return new HardSwishLayer(layer_name);
}

LayerRegisterWrapper HardSwishLayer_Register(createHardSwishInstance, "nn.Hardswish");

} // namespace layer
} // namespace inferx