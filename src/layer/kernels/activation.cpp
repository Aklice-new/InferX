#include "layer/kernels/activation.h"
#include "core/common.h"

namespace inferx
{
namespace layer
{

ActivationLayer::ActivationLayer(std::string layer_name)
    : Layer(layer_name)
{
}

StatusCode ActivationLayer::prepare_layer(
    const std::vector<Tensor::TensorPtr>& inputs, const std::vector<Tensor::TensorPtr>& outputs)
{
    this->inputs_ = inputs;
    this->outputs_ = outputs;
    for (auto input : this->inputs_)
    {
        if (input->raw_ptr() == nullptr)
        {
            input->apply_data();
        }
    }
    for (auto output : this->outputs_)
    {
        if (output->raw_ptr() == nullptr)
        {
            output->apply_data();
        }
    }
    return StatusCode::Success;
}

// 设置加载本层的参数
// StatusCode ActivationLayer::load_param(const std::map<std::string, pnnx::Parameter>& params) {

// }
// StatusCode ActivationLayer::load_model(const std::map<std::string, pnnx::Attribute>& attributes) {

// }

} // namespace layer
} // namespace inferx