#include "layer/kernels/silu.h"
#include "layer/layer_factory.h"

namespace inferx
{
namespace layer
{

SiluLayer::SiluLayer(std::string layer_name)
    : ActivationLayer(layer_name)
{
    activation_type_ = ActivationType::ActivationType_Silu;
}

Layer* createSiluInstance(std::string layer_name)
{
    return new SiluLayer(layer_name);
}

LayerRegisterWrapper SiluLayer_Register(createSiluInstance, "nn.Silu");

} // namespace layer
} // namespace inferx