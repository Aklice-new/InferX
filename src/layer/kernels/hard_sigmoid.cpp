#include "layer/kernels/hard_sigmoid.h"
#include "layer/layer_factory.h"

namespace inferx
{
namespace layer
{

HardSigmoidLayer::HardSigmoidLayer(std::string layer_name)
    : ActivationLayer(layer_name)
{
    activation_type_ = ActivationType::ActivationType_HardSigmoid;
}

Layer* createHardSigmoidInstance(std::string layer_name)
{
    return new HardSigmoidLayer(layer_name);
}

LayerRegisterWrapper HardSigmoidLayer_Register(createHardSigmoidInstance, "nn.Hardsigmoid");

} // namespace layer
} // namespace inferx