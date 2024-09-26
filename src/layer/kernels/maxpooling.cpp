#include "layer/kernels/maxpooling.h"
#include "layer/layer_factory.h"

#include <glog/logging.h>

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
    if (params.find("stride") == params.end())
    {
        LOG(ERROR) << "MaxPooling operator parameter is none, check your params.";
        return StatusCode::Failed;
    }
    stride_h_ = params.at("stride").ai[0];
    stride_w_ = params.at("stride").ai[1];

    if (params.find("padding") == params.end())
    {
        LOG(ERROR) << "MaxPooling operator parameter is none, check your params.";
        return StatusCode::Failed;
    }

    padding_h_ = params.at("padding").ai[0];
    padding_w_ = params.at("padding").ai[1];

    if (params.find("kernel_size") == params.end())
    {
        LOG(ERROR) << "MaxPooling operator parameter is none, check your params.";
        return StatusCode::Failed;
    }

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