#include "layer/kernels/tanh.h"
#include "layer/layer_factory.h"

namespace inferx
{
namespace layer
{

TanHLayer::TanHLayer(std::string layer_name)
    : ActivationLayer(layer_name)
{
    activation_type_ = ActivationType::ActivationType_TanH;
}

Layer* createTanHInstance(std::string layer_name)
{
    return new TanHLayer(layer_name);
}

LayerRegisterWrapper TanHLayer_Register(createTanHInstance, "nn.TanH");

} // namespace layer
} // namespace inferx