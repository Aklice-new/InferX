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

} // namespace layer
} // namespace inferx