#include "layer/kernels/adaptive_avgpooling.h"
#include "core/common.h"
#include "layer/layer_factory.h"

#include <glog/logging.h>

namespace inferx
{

namespace layer
{
AdaptiveAvgPoolingLayer::AdaptiveAvgPoolingLayer(std::string layer_name)
    : Layer(layer_name)
{
}

StatusCode AdaptiveAvgPoolingLayer::prepare_layer(
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

StatusCode AdaptiveAvgPoolingLayer::load_param(const std::map<std::string, pnnx::Parameter>& params)
{
    if (params.find("output_size") == params.end())
    {
        LOG(ERROR) << "AdaptiveAvgPooling operator parameter is none, check your params.";
        return StatusCode::Failed;
    }
    output_height_ = params.at("output_size").ai[0];
    output_width_ = params.at("output_size").ai[1];
    return StatusCode::Success;
}

Layer* createAdaptiveAvgPoolingInstance(std::string layer_name)
{
    return new AdaptiveAvgPoolingLayer(layer_name);
}

LayerRegisterWrapper AdaptiveAvgPoolingLayer_Register(
    createAdaptiveAvgPoolingInstance, "nn.AdaptiveAvgPool2d", "F.adaptive_avg_pool2d");

} // namespace layer
} // namespace inferx