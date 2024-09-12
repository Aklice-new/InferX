#include "layer/kernels/sigmoid.h"
#include "layer/layer_factory.h"

namespace inferx
{
namespace layer
{

SigmoidLayer::SigmoidLayer(std::string layer_name)
    : ActivationLayer(layer_name)
{
    activation_type_ = ActivationType::ActivationType_Sigmoid;
}

Layer* createSigmoidInstance(std::string layer_name)
{
    return new SigmoidLayer(layer_name);
}

LayerRegisterWrapper SigmoidLayer_Register(createSigmoidInstance, "nn.Sigmoid");

} // namespace layer
} // namespace inferx