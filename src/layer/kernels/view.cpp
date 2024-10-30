/**
 * @file view.cpp
 * @author Aklice (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2024-10-24
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "core/common.h"
#include "layer/kernels/view.h"
#include "layer/layer_factory.h"

#include <cstdint>
#include <glog/logging.h>

namespace inferx
{
namespace layer
{
ViewLayer::ViewLayer(std::string name)
    : Layer(name)
{
}
StatusCode ViewLayer::prepare_layer(
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
StatusCode ViewLayer::load_param(const std::map<std::string, pnnx::Parameter>& params)
{
    if (params.find("shape") == params.end())
    {
        LOG(ERROR) << "View operator param shape is none, check your model.";
        return StatusCode::Failed;
    }
    auto shape = params.at("shape").ai;
    for (auto s : shape)
    {
        shapes_.push_back(static_cast<uint32_t>(s));
    }
    return StatusCode::Success;
}
Layer* createViewInstance(std::string layer_name)
{
    return new ViewLayer(layer_name);
}

LayerRegisterWrapper ViewLayer_Register(createViewInstance, "Tensor.view");

} // namespace layer
} // namespace inferx