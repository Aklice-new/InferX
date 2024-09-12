#include "layer/kernels/maxpooling.h"
#include "layer/layer_factory.h"

namespace inferx
{
namespace layer
{

MaxPoolingLayer::MaxPoolingLayer(std::string name)
    : Layer(name)
{
}

StatusCode MaxPoolingLayer::prepare_layer(
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

StatusCode MaxPoolingLayer::load_param(const std::map<std::string, pnnx::Parameter>& params)
{
    stride_h_ = params.at("stride").ai[0];
    stride_w_ = params.at("stride").ai[1];
    padding_h_ = params.at("padding").ai[0];
    padding_w_ = params.at("padding").ai[1];
    pooling_size_h_ = params.at("kernel_size").ai[0];
    pooling_size_w_ = params.at("kernel_size").ai[1];
    return StatusCode::Success;
}

Layer* createMaxPoolingInstance(std::string layer_name)
{
    return new MaxPoolingLayer(layer_name);
}

LayerRegisterWrapper MaxPoolingLayerLayer_Register(createMaxPoolingInstance, "nn.MaxPool2d");
} // namespace layer
} // namespace inferx